#!/usr/bin/env python3
"""
tokgan_json_to_ae.py -- Tokgan JSON -> After Effects shape-layer importer.

Produces two files:
  1. A small After Effects .jsx loader script
  2. A sidecar .json data file (placed next to the .jsx)

The loader is small and parses fast; the data file is read by AE at runtime
via JSON.parse (with an eval fallback for ExtendScript builds that lack JSON).
Path keyframes are bulk-written with Property.setValuesAtTimes() which is
dramatically faster than per-frame setValueAtTime calls on long clips.

In AE:
    File > Scripts > Run Script File... > pick the .jsx
    (the sidecar .json must sit next to the .jsx)

The loader builds ONE shape layer ("Tokgan Shapes") containing one Vector
Group per source object. Each group has a Bezier path animated frame-by-frame
plus a hold-stepped opacity track driven by which source frames the object
exists in. Motion blur is enabled on the layer.

Timing: comp time 0 is the clip's FIRST frame (JSON frame "1"), for every
layer, whether or not anybody is in it yet (it used to be the first
frame with a person in it, so someone entering at frame 30 was keyed 29
frames early). A comp the loader creates runs at the JSON's fps. Older JSONs
stamped every file fps=24 whatever the footage ran at, so for those (no
`frame_start`, fps 24, no --fps) the loader still retimes the keys to the
active comp's frame rate, one source frame per comp frame, as it always has.
A JSON that carries `frame_start` (or any fps but 24) declares the real rate:
its keys go at their real times, and the loader says so when the active comp
runs at another rate. `frame_start` (the plate's own number for JSON frame 1,
e.g. 1001 or 977) also becomes the comp's displayStartFrame, so the timeline
reads in the plate's numbers.

--masks builds the AE-native "Solid + masks" layout instead: one solid
per person, one Add mask per body part, its Mask Path keyed per frame and its
Mask Opacity hold-stepped 0/100 for visibility. Much cheaper for After Effects
to build and evaluate than hundreds of keyed shape-layer groups. Flat only.

v3 JSONs (camera + persons blocks, Rotobot Next 0.10.0+) build a HIERARCHY
instead (pass --flat for the single layer above):

    main comp
      "Tokgan Shapes"  -- a precomp layer with a Corner Pin keyed every frame:
                          the plate camera, as the exact ECC homography
        "Tokgan Stabilised" precomp (camera removed, padded to fit):
          p<N>_pelvis  -- Null per person, position keyed every frame
            <object>   -- shape layer per body part, parented to its person,
                          position + rotation keyed from the bone; the path
                          is in bone-local pixels

so an artist can stabilise, move a whole person, or adjust one limb without
touching every vertex. The decomposition comes from rotobot-nuke's
rotobot_nuke.hierarchy, vendored beside this script as _rotobot_hierarchy/
(nothing to install); the composed result is checked against every source
vertex before anything is written.

Every loader shows its progress in a non-modal palette and logs each step,
timestamped, to <stem>_import.log next to the .jsx. Only a failure opens a
dialog; a timing note (the comp runs at another fps) goes in the palette and
the comp's comment. Path keys are capped at --max-cp control points (default
96), resampled evenly along the curve, which keeps dense shapes quick to
import.

Use `python3 tokgan_json_to_ae.py --help` for invocation details.

Coordinate convention:
    Both the JSON and AE use Y-down, origin top-left pixel coordinates.
    Bezier handles in JSON are absolute (left_x/left_y, right_x/right_y);
    AE expects them as deltas from the vertex, so we subtract.
"""

import colorsys
import io
import json
import os
import sys
from typing import NamedTuple


# Persons whose maximum-over-time bounding-box max dimension is smaller
# than max(comp_w, comp_h) / BACKGROUND_DIM_DIVISOR are treated as
# background figures: by default dropped from the import; with
# --keep-background they are imported but share one neutral-grey fill.
BACKGROUND_DIM_DIVISOR = 20

# Bump when the sidecar payload schema changes in a way the loader
# would have to handle. Sidecars whose stored value differs from this
# constant are rebuilt on the next run regardless of mtime, so older
# sidecars without `color` don't quietly stay around.
PAYLOAD_SCHEMA = 4  # 4: retime / frame_start, clip-start time zero, masks mode

# The layouts --masks / layout= choose between. "shapes" is the shape-layer
# import (hierarchy for v3, else one flat layer); "masks" is Solid + masks.
LAYOUTS = ("shapes", "masks")

# JSON pixel coordinates name pixel CENTRES; AE comp space puts (0, 0) at the
# first pixel's top-left CORNER. The shape loaders move their top layer to
# (0.5, 0.5); the masks loader bakes the same half pixel into every vertex
# instead, because moving a comp-sized solid would leave a half-pixel strip
# at the top and left edge that no mask can cover.
PIXEL_CENTRE = 0.5

# Hierarchy mode: the stabilised precomp is padded this far beyond the
# stabilised extent of every vertex, and must stay within AE's comp limit.
PRECOMP_MARGIN = 64
AE_MAX_COMP_SIZE = 30000

BACKGROUND_GREY = [0.5, 0.5, 0.5]

# At most this many control points per path key (--max-cp; 0 = no cap). Every
# point is keyed on every frame, so the sidecar, the loader's parse and AE's
# keying all grow with it: dense 4K shapes reach ~180 points per key, and a
# long shot then takes minutes to import. Shapes over the cap are resampled
# evenly by arc length (resample_points); the source JSON is never touched.
DEFAULT_MAX_CP = 96

# Arc length is measured on each Bezier segment split this many times: plenty
# for segments a few pixels long, and it keeps the cap cheap in plain Python.
ARC_SUBDIV = 4


def round_pt(x, y, ndigits=2):
    return [round(x, ndigits), round(y, ndigits)]


def _bez(p0, p1, p2, p3, t):
    """Point and derivative of the cubic p0..p3 at t, as (x, y, dx, dy)."""
    u = 1.0 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    da, db, dc, dd = -3 * u * u, 3 * u * u - 6 * u * t, 6 * u * t - 3 * t * t, 3 * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            da * p0[0] + db * p1[0] + dc * p2[0] + dd * p3[0],
            da * p0[1] + db * p1[1] + dc * p2[1] + dd * p3[1])


def resample_points(pts, closed, max_cp):
    """`pts` (JSON point dicts: x, y and optional absolute left_/right_
    handles) cut to `max_cp` knots spaced evenly by arc length along the
    curve they draw, starting at the first knot. A closed shape wraps from
    its last knot to its first, as build_shape_record keys it; an open one
    keeps both ends. Handles lie along the curve's tangent, a third of the
    neighbouring spacing long; a shape without handles stays a polygon.
    Returns `pts` itself when it is already within the cap (or max_cp is 0)."""
    n = len(pts)
    if not max_cp or n <= max_cp or n < 3:
        return pts
    max_cp = max(max_cp, 3 if closed else 2)
    smooth = any(("left_x" in p and (p["left_x"] != p["x"] or p["left_y"] != p["y"])) or
                 ("right_x" in p and (p["right_x"] != p["x"] or p["right_y"] != p["y"]))
                 for p in pts)

    segs = []  # (p0, p1, p2, p3) per Bezier segment
    for i in range(n if closed else n - 1):
        a, b = pts[i], pts[(i + 1) % n]
        p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])
        p1 = (a.get("right_x", p0[0]), a.get("right_y", p0[1]))
        p2 = (b.get("left_x", p3[0]), b.get("left_y", p3[1]))
        segs.append((p0, p1, p2, p3))

    # Cumulative arc length at every subdivision: (length, segment, t).
    table = [(0.0, 0, 0.0)]
    total = 0.0
    for si, (p0, p1, p2, p3) in enumerate(segs):
        px, py = p0
        for k in range(1, ARC_SUBDIV + 1):
            t = k / ARC_SUBDIV
            x, y, _, _ = _bez(p0, p1, p2, p3, t)
            total += ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            table.append((total, si, t))
            px, py = x, y
    if total <= 0.0:
        return pts[:max_cp]

    step = total / (max_cp if closed else max_cp - 1)
    out = []
    j = 0
    for k in range(max_cp):
        s = min(k * step, total)
        while j + 1 < len(table) - 1 and table[j + 1][0] < s:
            j += 1
        (s0, si0, t0), (s1, si1, t1) = table[j], table[j + 1]
        if si1 != si0:          # the end of segment si0 is t = 0 of si1
            t0 = 0.0
        f = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
        t = t0 + (t1 - t0) * f
        x, y, dx, dy = _bez(*segs[si1], t)
        rec = {"x": x, "y": y}
        dl = (dx * dx + dy * dy) ** 0.5
        if smooth and dl > 0.0:
            h = step / 3.0 / dl
            rec.update(left_x=x - dx * h, left_y=y - dy * h,
                       right_x=x + dx * h, right_y=y + dy * h)
        else:
            rec.update(left_x=x, left_y=y, right_x=x, right_y=y)
        out.append(rec)
    if not closed:
        # Exact ends, and no handle beyond them.
        for rec, src in ((out[0], pts[0]), (out[-1], pts[-1])):
            ox, oy = src["x"] - rec["x"], src["y"] - rec["y"]
            for kx, ky in (("x", "y"), ("left_x", "left_y"), ("right_x", "right_y")):
                rec[kx] += ox
                rec[ky] += oy
        out[0]["left_x"], out[0]["left_y"] = out[0]["x"], out[0]["y"]
        out[-1]["right_x"], out[-1]["right_y"] = out[-1]["x"], out[-1]["y"]
    return out


def cap_control_points(data, max_cp):
    """Resample every path key in the raw JSON `data` (in place) to at most
    `max_cp` knots. Returns how many keys were resampled."""
    if not max_cp:
        return 0
    n = 0
    for obj in data["objects"].values():
        closed = bool(obj.get("closed", True))
        for fr in obj.get("frames", {}).values():
            pts = fr.get("points") if isinstance(fr, dict) else None
            if pts and len(pts) > max_cp:
                fr["points"] = resample_points(pts, closed, max_cp)
                n += 1
    return n


def person_id_of(obj_name, obj):
    """Canonical int person id. Prefer obj['person_id']; fall back to
    parsing the 'pK' prefix in the object's name."""
    pid = obj.get("person_id")
    if pid is not None:
        return int(pid)
    prefix = obj_name.split(":", 1)[0]
    return int(prefix.lstrip("p"))


def hue_palette(n):
    """n evenly spaced hues at full saturation/value, as 3-float RGB
    lists in 0..1. n==0 returns []. Hue 0 (pure red) is always present
    when n>=1 so the single-person sample renders identically to today."""
    if n <= 0:
        return []
    return [
        [round(c, 6) for c in colorsys.hsv_to_rgb(i / n, 1.0, 1.0)]
        for i in range(n)
    ]


def classify_persons(objects, width, height):
    """Return (person_color, foreground_ids, background_ids).

    person_color maps int person_id -> [r,g,b] (foreground hue or
    BACKGROUND_GREY). foreground_ids and background_ids are sorted
    lists of ints. Classification uses the per-person maximum across
    all frames of max(bbox_w, bbox_h); the threshold is
    max(width, height) / BACKGROUND_DIM_DIVISOR.
    """
    threshold = max(width, height) / BACKGROUND_DIM_DIVISOR

    # (pid, frame) -> [minx, maxx, miny, maxy]
    per_pf = {}
    for obj_name, obj in objects.items():
        pid = person_id_of(obj_name, obj)
        for fkey, fr in obj.get("frames", {}).items():
            pts = fr.get("points")
            if not pts:
                continue
            key = (pid, int(fkey))
            box = per_pf.get(key)
            for p in pts:
                x, y = p["x"], p["y"]
                if box is None:
                    box = [x, x, y, y]
                    per_pf[key] = box
                else:
                    if x < box[0]: box[0] = x
                    if x > box[1]: box[1] = x
                    if y < box[2]: box[2] = y
                    if y > box[3]: box[3] = y

    max_size = {}
    for (pid, _f), box in per_pf.items():
        sz = max(box[1] - box[0], box[3] - box[2])
        if sz > max_size.get(pid, 0.0):
            max_size[pid] = sz

    foreground_ids = sorted(p for p, s in max_size.items() if s >= threshold)
    background_ids = sorted(p for p, s in max_size.items() if s < threshold)

    palette = hue_palette(len(foreground_ids))
    person_color = {pid: palette[i] for i, pid in enumerate(foreground_ids)}
    for pid in background_ids:
        person_color[pid] = BACKGROUND_GREY

    return person_color, foreground_ids, background_ids


def build_shape_record(obj_name, obj, height, start_frame, end_frame, fps):
    """Produce a compact dict: name, closed, times, verts, inT, outT, opacity.

    JSON coords are screen-style (Y-down, origin top-left) -- same as AE --
    despite an upstream docstring claiming Nuke convention. No Y flip.
    """
    times, verts_all, in_all, out_all = [], [], [], []
    path_frames = []  # source frame numbers that produced a key
    for fkey in sorted(obj.get("frames", {}).keys(), key=int):
        pts = obj["frames"][fkey].get("points", [])
        if not pts:
            continue
        verts, in_t, out_t = [], [], []
        for p in pts:
            x, y = p["x"], p["y"]
            verts.append(round_pt(x, y))
            if "left_x" in p:
                in_t.append(round_pt(p["left_x"] - x, p["left_y"] - y))
            else:
                in_t.append([0.0, 0.0])
            if "right_x" in p:
                out_t.append(round_pt(p["right_x"] - x, p["right_y"] - y))
            else:
                out_t.append([0.0, 0.0])
        t = (int(fkey) - start_frame) / fps
        times.append(round(t, 6))
        verts_all.append(verts)
        in_all.append(in_t)
        out_all.append(out_t)
        path_frames.append(int(fkey))

    # Opacity: drive 0 when the shape doesn't exist in a source frame, 100
    # when it does. The shape "exists" iff visibility[f]==1 (preferred) or
    # iff f has path data (fallback when visibility is absent). The keys are
    # held, so the minimal set is one at the clip's start and one wherever
    # the value changes: a part that flickers costs two keys per gap, and
    # one that is always there costs one.
    visibility = obj.get("visibility", {})
    if visibility:
        exists = set(int(k) for k, v in visibility.items() if v)
    else:
        exists = set(path_frames)

    opacity = []
    if exists:
        last = None
        for f in range(start_frame, end_frame + 1):
            v = 100 if f in exists else 0
            if v != last:
                opacity.append([round((f - start_frame) / fps, 6), v])
                last = v

    return {
        "name": obj_name,
        "closed": bool(obj.get("closed", True)),
        "times": times,
        "verts": verts_all,
        "inT": in_all,
        "outT": out_all,
        "opacity": opacity,
    } if times else None


def _r(v, nd=3):
    return round(v, nd)


def _hierarchy_api():
    """(load_json, hierarchy module) for the v3 decomposition.

    The vendored copy beside this script first (_rotobot_hierarchy/, a
    byte-for-byte copy of rotobot-nuke's reader + hierarchy at the commit in
    its VENDORED.txt), so the import needs nothing installed and always uses
    the tested version; an installed rotobot-nuke otherwise.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from _rotobot_hierarchy import hierarchy, reader
        return reader.load_json, hierarchy
    except ImportError:
        pass
    try:
        from rotobot_nuke import hierarchy, load_json
        return load_json, hierarchy
    except ImportError:
        raise SystemExit(
            "This JSON has camera/person data (schema v3), and the hierarchy "
            "code is missing: keep the _rotobot_hierarchy folder next to "
            "tokgan_json_to_ae.py (or pip install rotobot-nuke), or pass "
            "--flat for the single-layer import.")


def build_hierarchy_payload(in_path, data, keep_names, person_color,
                            width, height, fps, start_frame, end_frame):
    """Payload for the hierarchy loader: camera corner pin, person Nulls and
    bone-local shape layers. Raises SystemExit with a fix when the hierarchy
    code is missing or the decomposition does not reproduce the source."""
    load_json, hier = _hierarchy_api()
    ROUND_TRIP_TOLERANCE_PX = hier.ROUND_TRIP_TOLERANCE_PX
    apply_h, decompose, round_trip_error = hier.apply_h, hier.decompose, hier.round_trip_error

    # `in_path` is a path, or a text stream of the (control-point capped) JSON.
    doc = load_json(in_path)
    doc.objects = {k: o for k, o in doc.objects.items() if k in keep_names}
    hier = decompose(doc)
    err = round_trip_error(doc, hier)
    if err > ROUND_TRIP_TOLERANCE_PX:
        raise SystemExit(
            "hierarchy does not reproduce the source vertices (worst error "
            "%.3f px > %.2f px); use --flat for this file" % (err, ROUND_TRIP_TOLERANCE_PX))

    # Stabilised extent of everything the precomp has to hold: every vertex
    # and handle, plus the seed plate itself.
    xs, ys = [0.0, float(width)], [0.0, float(height)]
    for key, obj in doc.objects.items():
        for f, fr in obj.frames.items():
            inv = hier.camera_inv[f]
            for p in fr.points:
                for x, y in ((p.x, p.y), (p.left_x, p.left_y), (p.right_x, p.right_y)):
                    sx, sy = apply_h(inv, x, y)
                    xs.append(sx); ys.append(sy)
    ox = PRECOMP_MARGIN - int(min(xs) // 1)
    oy = PRECOMP_MARGIN - int(min(ys) // 1)
    pw = int(max(xs) // 1) + 1 + ox + PRECOMP_MARGIN
    ph = int(max(ys) // 1) + 1 + oy + PRECOMP_MARGIN
    if max(pw, ph) > AE_MAX_COMP_SIZE:
        raise SystemExit(
            "the stabilised precomp would be %dx%d, beyond After Effects' "
            "%d px limit (a very long camera move); use --flat" % (pw, ph, AE_MAX_COMP_SIZE))

    def t_of(f):
        return round((f - start_frame) / fps, 6)

    # Corner Pin, AE's order: upper-left, upper-right, lower-left, lower-right
    # of the precomp layer, each mapped through the frame's homography.
    cam_times, cam_corners = [], []
    for f in hier.frames:
        h = hier.camera[f]
        cam_times.append(t_of(f))
        cam_corners.append([
            [_r(c) for c in apply_h(h, cx - ox, cy - oy)]
            for cx, cy in ((0, 0), (pw, 0), (0, ph), (pw, ph))
        ])

    persons = []
    for pid in sorted({hier.person_of[k] for k in doc.objects}):
        series = hier.pelvis[pid]
        persons.append({
            "pid": pid,
            "name": "p%d_pelvis" % pid,
            "times": [t_of(f) for f in hier.frames],
            "pos": [[_r(series[f][0] + ox), _r(series[f][1] + oy)] for f in hier.frames],
        })

    shapes = []
    for key, obj in doc.objects.items():
        raw_obj = data["objects"][key]
        flat = build_shape_record(key, raw_obj, height, start_frame, end_frame, fps)
        if not flat:
            continue
        times, verts, in_t, out_t, xf_pos, xf_rot = [], [], [], [], [], []
        for f in sorted(obj.frames):
            pts = hier.knots[key][f]
            if not pts:
                continue
            part = hier.parts[key][f]
            times.append(t_of(f))
            verts.append([[_r(q.x), _r(q.y)] for q in pts])
            in_t.append([[_r(q.left_x - q.x), _r(q.left_y - q.y)] for q in pts])
            out_t.append([[_r(q.right_x - q.x), _r(q.right_y - q.y)] for q in pts])
            xf_pos.append([_r(part.tx), _r(part.ty)])
            xf_rot.append(_r(part.angle, 5))
        shapes.append({
            "name": key,
            "pid": hier.person_of[key],
            "closed": flat["closed"],
            "color": person_color[hier.person_of[key]],
            "times": times,
            "verts": verts,
            "inT": in_t,
            "outT": out_t,
            "xf": {"pos": xf_pos, "rot": xf_rot},
            "opacity": flat["opacity"],
        })

    return {
        "schema": PAYLOAD_SCHEMA,
        "mode": "hierarchy",
        "width": width,
        "height": height,
        "fps": fps,
        "duration": (end_frame - start_frame + 1) / fps,
        "precomp": {"width": pw, "height": ph, "offset": [ox, oy]},
        "camera": {"times": cam_times, "corners": cam_corners},
        "persons": persons,
        "shapes": shapes,
        "held": hier.held,
    }


# ExtendScript shared by all three loaders, pasted into each (a loader is one
# self-contained .jsx). Plain ES3: no let/const, no arrow functions. Braces are
# literal here: it is substituted INTO the templates, never formatted itself.
COMMON_JS = r"""
        // How to turn the sidecar's times (seconds from the clip's first
        // frame, at data.fps) into this comp's times.
        //
        // data.retime (older JSONs, which stamped every file fps=24 whatever
        // the footage ran at): scale so each source frame lands on one comp
        // frame, the long-standing behaviour. Otherwise data.fps is the
        // plate's real rate, so the keys stay at their real times -- where
        // the plate's own frames are in a comp of any rate -- and when this
        // comp runs at another rate the operator is told rather than
        // silently re-timed. A sidecar from before `retime` existed is old,
        // so it is retimed.
        function tokganTiming(comp, data) {
            if (data.retime !== false) {
                return { scale: data.fps / comp.frameRate, note: "" };
            }
            var note = "";
            if (Math.abs(comp.frameRate - data.fps) > 0.001) {
                note = "This clip runs at " + data.fps + " fps; the comp \"" + comp.name +
                       "\" runs at " + comp.frameRate + " fps. The keys are at the clip's own " +
                       "times, so some fall between this comp's frames. Set the comp to " +
                       data.fps + " fps (Composition > Composition Settings) for one key per frame.";
            }
            return { scale: 1, note: note };
        }

        // The plate's own frame numbers on the timeline (1001, 977...): time 0
        // stays the clip's first frame, only the display changes. Left alone
        // when the JSON does not say (data.frame_start absent). displayStartFrame
        // is AE 17.1+; earlier builds only have displayStartTime, in seconds.
        function tokganDisplayStart(comp, data) {
            if (data.frame_start === undefined || data.frame_start === null) return;
            try {
                if (typeof comp.displayStartFrame === "number") {
                    comp.displayStartFrame = data.frame_start;
                } else {
                    comp.displayStartTime = data.frame_start / comp.frameRate;
                }
            } catch (eDs) {
                $.writeln("[Tokgan] could not set the display start frame: " + eDs.toString());
            }
        }

        // Progress: a non-modal palette, and a log next to the .jsx
        // (<stem>_import.log) with one timestamped line per STEP, so an import
        // that seems stuck says where it is. Nothing on the success path is
        // modal: a modal alert can open behind the host window on Windows, and
        // After Effects then looks frozen. Only a failure gets a dialog. The
        // palette and the log never stop an import: every call is wrapped, and
        // one that cannot be made (no UI, a read-only folder) is left out.
        function tokganPad(n) { return (n < 10 ? "0" : "") + n; }
        function tokganStamp() {
            var d = new Date();
            return d.getFullYear() + "-" + tokganPad(d.getMonth() + 1) + "-" + tokganPad(d.getDate()) +
                   " " + tokganPad(d.getHours()) + ":" + tokganPad(d.getMinutes()) + ":" +
                   tokganPad(d.getSeconds());
        }
        function tokganElapsed(pr) { return ((new Date()).getTime() - pr.t0) / 1000; }
        function tokganLog(pr, line) {
            $.writeln("[Tokgan] " + line);
            if (!pr.log) return;
            try {
                // Opened and closed per line, so the log is complete up to
                // the moment After Effects stopped, whatever stopped it.
                if (pr.log.open("a")) { pr.log.writeln(line); pr.log.close(); }
            } catch (eLg) { pr.log = null; }
        }
        function tokganShow(pr, text) {
            if (!pr.text) return;
            try {
                pr.text.text = text;
                pr.win.update();
            } catch (eSh) { pr.text = null; }
        }
        function tokganProgressOpen(title) {
            var pr = { t0: (new Date()).getTime(), title: title, win: null, text: null, log: null };
            try {
                pr.log = new File((new File($.fileName)).fsName.replace(/\.jsx(bin)?$/i, "") + "_import.log");
                pr.log.encoding = "UTF-8";
                if (pr.log.open("w")) {
                    pr.log.writeln("[" + tokganStamp() + "] " + title + ": " + $.fileName);
                    pr.log.writeln("After Effects " + app.version + "; native JSON: " +
                                   (typeof JSON !== "undefined" && JSON.parse ? "yes" : "no (eval)"));
                    pr.log.close();
                } else {
                    pr.log = null;
                }
            } catch (eLo) { pr.log = null; }
            try {
                pr.win = new Window("palette", title, undefined, { resizeable: true });
                pr.win.orientation = "column";
                pr.win.alignChildren = ["fill", "top"];
                pr.text = pr.win.add("statictext", undefined, "Starting...", { multiline: true });
                pr.text.preferredSize = [520, 110];
                // A Function built from source sees only the global scope. A
                // closure here would keep the loader's scope -- the whole
                // parsed sidecar -- alive for as long as the palette is open.
                pr.win.add("button", undefined, "Close").onClick = new Function("this.window.close();");
                pr.win.show();
            } catch (eWi) { pr.win = null; pr.text = null; }
            return pr;
        }
        // Record a STEP (with counts in its text) and return it for STEP.
        function tokganStep(pr, step) {
            var t = tokganElapsed(pr).toFixed(1);
            tokganLog(pr, "[" + tokganStamp() + " +" + t + "s] " + step);
            tokganShow(pr, pr.title + "\n" + step + "\n" + t + " s");
            return step;
        }
        // The end of an import: the summary (and any note) stays in the
        // palette until the operator closes it.
        function tokganDone(pr, summary, note) {
            var t = tokganElapsed(pr).toFixed(1);
            tokganLog(pr, "[" + tokganStamp() + " +" + t + "s] done: " + summary + " in " + t + " s");
            if (note) tokganLog(pr, "note: " + note);
            tokganShow(pr, summary + " in " + t + " s" + (note ? "\n\n" + note : ""));
        }
        function tokganFailed(pr, step, err) {
            var t = tokganElapsed(pr).toFixed(1);
            tokganLog(pr, "[" + tokganStamp() + " +" + t + "s] FAILED at " + step + ": " + err.toString() +
                          " (line " + (err.line || "?") + ")");
            tokganShow(pr, "Import failed at: " + step + "\n" + err.toString());
        }

        // The timing note goes in the comp's comment (replacing an earlier
        // import's), where it stays with the comp, rather than in a dialog.
        function tokganCommentNote(comp, note) {
            var PREFIX = "Tokgan import: ";
            try {
                var lines = String(comp.comment || "").split("\n"), keep = [];
                for (var i = 0; i < lines.length; i++) {
                    if (lines[i].indexOf(PREFIX) !== 0) keep.push(lines[i]);
                }
                while (keep.length && keep[keep.length - 1] === "") keep.pop();
                if (note) keep.push(PREFIX + note);
                comp.comment = keep.join("\n");
            } catch (eCm) {
                $.writeln("[Tokgan] could not set the comp comment: " + eCm.toString());
            }
        }

        // The sidecar: by path first (no folder listing), then by listing
        // the folder (robust to ':' and spaces in names), then by asking.
        function tokganFindSidecar(expectedName) {
            var folder = (new File($.fileName)).parent;
            try {
                var direct = new File(folder.fsName + "/" + expectedName);
                if (direct.exists) return direct;
            } catch (eDi) {}
            var siblings = folder.getFiles();
            for (var fi = 0; fi < siblings.length; fi++) {
                if (siblings[fi] instanceof File && siblings[fi].displayName === expectedName) {
                    return siblings[fi];
                }
            }
            // The .jsx alone is useless: the sidecar must travel with it.
            return File.openDialog(
                "Locate sidecar: " + expectedName +
                " (generated by tokgan_json_to_ae.py, NOT the raw Tokgan JSON)");
        }

        // Memory: a session that imports large sidecars one after another
        // does not get its memory back unless the loader lets go. Each loader
        // drops the raw text once parsed, each part's data once keyed, and
        // the whole sidecar before the summary, then asks for a collection.
        function tokganGc() {
            try { $.gc(); } catch (eGc) {}
        }

        // Hold-stepped keys from [[seconds, value], ...] (visibility) in one
        // call, then every key set to HOLD.
        function tokganHeldKeys(prop, keys, scale) {
            var ts = new Array(keys.length), vs = new Array(keys.length);
            for (var i = 0; i < keys.length; i++) {
                ts[i] = keys[i][0] * scale;
                vs[i] = keys[i][1];
            }
            prop.setValuesAtTimes(ts, vs);
            for (var k = 1; k <= prop.numKeys; k++) {
                prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD,
                                               KeyframeInterpolationType.HOLD);
            }
        }

        // JSON.parse where the host has it; ExtendScript (After Effects 2025
        // included) has no native JSON, so eval inside parens -- safe here
        // because the sidecar is written by tokgan_json_to_ae.py.
        function tokganParse(raw) {
            if (typeof JSON !== "undefined" && JSON.parse) return JSON.parse(raw);
            return eval("(" + raw + ")");
        }
"""

assert COMMON_JS.isascii(), "COMMON_JS must be pure ASCII"


LOADER_TEMPLATE = r"""// Auto-generated AE loader for Tokgan shape data.
// Sidecar JSON ({data_basename}) is expected next to this script.
(function() {{
{common}
    var STEP = "init";
    var prog = tokganProgressOpen("Tokgan import");
    try {{
        var expectedName = "{data_basename}";

        STEP = tokganStep(prog, "find-sidecar (" + expectedName + ")");
        var dataFile = tokganFindSidecar(expectedName);
        if (!dataFile) {{
            tokganStep(prog, "cancelled: no sidecar");
            return;
        }}

        STEP = tokganStep(prog, "read-sidecar (" + dataFile.fsName + ", " + dataFile.length + " bytes)");
        dataFile.encoding = "UTF-8";
        if (!dataFile.open("r")) {{
            throw new Error("File.open returned false for " + dataFile.fsName);
        }}
        var raw = dataFile.read();
        dataFile.close();
        if (!raw || raw.length === 0) {{
            throw new Error("File.read returned empty string");
        }}

        STEP = tokganStep(prog, "json-parse (" + raw.length + " chars)");
        var t0 = (new Date()).getTime();
        var data = tokganParse(raw);
        raw = null;
        var tParse = ((new Date()).getTime() - t0) / 1000;
        if (!data.shapes || data.duration === undefined) {{
            throw new Error(
                dataFile.displayName + " is not a Tokgan sidecar (no shapes/duration). " +
                "If it is the raw Tokgan JSON, run tokgan_json_to_ae.py on it and " +
                "keep the generated " + expectedName + " next to this .jsx."
            );
        }}

        STEP = tokganStep(prog, "init-comp (parsed in " + tParse.toFixed(1) + " s)");
        app.beginUndoGroup("Import Tokgan Shapes");

        var comp = app.project.activeItem;
        if (!(comp instanceof CompItem)) {{
            comp = app.project.items.addComp(
                "TokganShapes",
                data.width, data.height, 1,
                data.duration, data.fps
            );
        }}
        comp.motionBlur = true;
        tokganDisplayStart(comp, data);
        var timing = tokganTiming(comp, data);
        var tBuild = (new Date()).getTime();

        function mkShape(v, i, o, c) {{
            var s = new Shape();
            s.vertices = v;
            s.inTangents = i;
            s.outTangents = o;
            s.closed = c;
            return s;
        }}

        function setPathKeys(pathProp, times, shapes) {{
            try {{
                pathProp.setValuesAtTimes(times, shapes);
            }} catch (e) {{
                for (var i = 0; i < times.length; i++) {{
                    pathProp.setValueAtTime(times[i], shapes[i]);
                }}
            }}
        }}

        STEP = tokganStep(prog, "create-shape-layer");
        // Remove a previous "Tokgan Shapes" layer so re-runs replace cleanly.
        for (var rl = comp.numLayers; rl >= 1; rl--) {{
            if (comp.layer(rl).name === "Tokgan Shapes") comp.layer(rl).remove();
        }}
        // Single shape layer holding every imported path as a sub-group.
        var layer = comp.layers.addShape();
        layer.name = "Tokgan Shapes";
        var root = layer.property("ADBE Root Vectors Group");

        // Sidecar seconds -> comp seconds: 1 when the JSON's fps is real,
        // else re-timed to one source frame per comp frame (tokganTiming).
        var fpsScale = timing.scale;
        $.writeln("[Tokgan] data fps=" + data.fps + " comp fps=" + comp.frameRate + " scale=" + fpsScale);

        var shapes = data.shapes;
        for (var s = 0; s < shapes.length; s++) {{
            var sd = shapes[s];
            STEP = tokganStep(prog, "group " + (s + 1) + "/" + shapes.length + " (" + sd.name + ", " +
                                    sd.times.length + " keys)");

            var grp = root.addProperty("ADBE Vector Group");
            grp.name = sd.name;
            var inside = grp.property("ADBE Vectors Group");

            // AE 25.6 invalidates earlier property references when subsequent
            // addProperty calls mutate the parent group. Do ALL the addProperty
            // calls first, then walk the contents to (re)acquire the live
            // pathItem reference and from it the bezier Path child.
            inside.addProperty("ADBE Vector Shape - Group");
            inside.addProperty("ADBE Vector Graphic - Fill");
            var pathItem = null;
            var fillItem = null;
            for (var pp = 1; pp <= inside.numProperties; pp++) {{
                var mn = inside.property(pp).matchName;
                if (mn === "ADBE Vector Shape - Group") {{
                    pathItem = inside.property(pp);
                }} else if (mn === "ADBE Vector Graphic - Fill") {{
                    fillItem = inside.property(pp);
                }}
            }}
            var pathProp = pathItem.property("ADBE Vector Shape");
            if (fillItem && sd.color) {{
                fillItem.property("ADBE Vector Fill Color").setValue(sd.color);
            }}

            var shapeObjs = new Array(sd.times.length);
            var scaledTimes = new Array(sd.times.length);
            for (var i = 0; i < sd.times.length; i++) {{
                shapeObjs[i] = mkShape(sd.verts[i], sd.inT[i], sd.outT[i], sd.closed);
                scaledTimes[i] = sd.times[i] * fpsScale;
            }}

            setPathKeys(pathProp, scaledTimes, shapeObjs);

            // Per-group opacity for visibility -- lives under the group's
            // Transform, NOT layer.transform (which would affect everything).
            if (sd.opacity.length > 0) {{
                var op = grp.property("ADBE Vector Transform Group").property("ADBE Vector Group Opacity");
                tokganHeldKeys(op, sd.opacity, fpsScale);
            }}
            // Keyed: release this part's data.
            shapeObjs = null; scaledTimes = null; sd = null; shapes[s] = null;
        }}

        STEP = tokganStep(prog, "layer-transform-and-motion-blur");
        // Layer origin at comp (0.5, 0.5): the JSON's pixel coordinates name pixel CENTRES (as Silhouette,
        // Nuke and the Kthanid matte read them), while AE's comp space puts (0, 0) at the top-left
        // CORNER of the first pixel. At (0, 0) every shape rendered half a pixel up and left -- measured
        // against the matte through AE 2025's own render (5362065, 2026-10-17).
        layer.transform.anchorPoint.setValue([0, 0]);
        layer.transform.position.setValue([0.5, 0.5]);
        layer.motionBlur = true;

        tokganCommentNote(comp, timing.note);
        var tDone = ((new Date()).getTime() - tBuild) / 1000;
        app.endUndoGroup();

        var summary = "Imported " + shapes.length + " shape groups (JSON parse " + tParse.toFixed(1) +
                      " s, AE build " + tDone.toFixed(1) + " s)";
        data = null; shapes = null;
        tokganGc();
        tokganDone(prog, summary, timing.note);
    }} catch (err) {{
        try {{ app.endUndoGroup(); }} catch (e2) {{}}
        raw = null; data = null; shapes = null;
        tokganGc();
        tokganFailed(prog, STEP, err);
        // The one modal dialog: a failure, after the undo group is closed.
        alert(
            "Tokgan import failed.\n" +
            "Step: " + STEP + "\n" +
            "Error: " + err.toString() + "\n" +
            "Line:  " + (err.line || "?") + "\n" +
            "File:  " + (err.fileName || "?")
        );
    }}
}})();
"""

# The loader is written verbatim into the .jsx. Keep it ASCII so the output
# is byte-identical on every platform: ExtendScript has no reliable way to
# tell UTF-8 from cp1252, and Windows' default codec mangles e.g. em dashes.
assert LOADER_TEMPLATE.isascii(), "LOADER_TEMPLATE must be pure ASCII"


# Hierarchy loader (v3 JSONs). Kept separate from LOADER_TEMPLATE so the flat
# import is untouched. Plain ExtendScript (ES3): no let, no arrow functions.
# __DATA_BASENAME__ is substituted with str.replace, so braces need no escaping.
LOADER_HIER_TEMPLATE = r"""// Auto-generated AE loader for Tokgan shape data (camera / person hierarchy).
// Sidecar JSON (__DATA_BASENAME__) is expected next to this script.
(function() {
__COMMON__
    var STEP = "init";
    var prog = tokganProgressOpen("Tokgan import (hierarchy)");
    try {
        var expectedName = "__DATA_BASENAME__";
        STEP = tokganStep(prog, "find-sidecar (" + expectedName + ")");
        var dataFile = tokganFindSidecar(expectedName);
        if (!dataFile) {
            tokganStep(prog, "cancelled: no sidecar");
            return;
        }

        STEP = tokganStep(prog, "read-sidecar (" + dataFile.fsName + ", " + dataFile.length + " bytes)");
        dataFile.encoding = "UTF-8";
        if (!dataFile.open("r")) throw new Error("File.open returned false for " + dataFile.fsName);
        var raw = dataFile.read();
        dataFile.close();

        STEP = tokganStep(prog, "json-parse (" + raw.length + " chars)");
        var data = tokganParse(raw);
        raw = null;
        if (data.mode !== "hierarchy" || !data.camera || !data.precomp) {
            throw new Error(dataFile.displayName + " is not a Tokgan hierarchy sidecar; " +
                            "re-run tokgan_json_to_ae.py on the raw JSON.");
        }

        app.beginUndoGroup("Import Tokgan Hierarchy");

        STEP = tokganStep(prog, "init-comp");
        var comp = app.project.activeItem;
        if (!(comp instanceof CompItem)) {
            comp = app.project.items.addComp("TokganShapes", data.width, data.height, 1,
                                             data.duration, data.fps);
        }
        comp.motionBlur = true;
        tokganDisplayStart(comp, data);
        var timing = tokganTiming(comp, data);
        var fpsScale = timing.scale;

        // Re-runs replace cleanly: the layer in this comp, then the precomp.
        for (var rl = comp.numLayers; rl >= 1; rl--) {
            if (comp.layer(rl).name === "Tokgan Shapes") comp.layer(rl).remove();
        }
        for (var ri = app.project.numItems; ri >= 1; ri--) {
            var it = app.project.item(ri);
            if (it instanceof CompItem && it.name === "Tokgan Stabilised") it.remove();
        }

        function scaled(times) {
            var out = new Array(times.length);
            for (var i = 0; i < times.length; i++) out[i] = times[i] * fpsScale;
            return out;
        }
        // Linear in time, and straight lines in space for position: the
        // undersampler measures error against linear interpolation.
        function keyLinear(prop, times, values) {
            try { prop.setValuesAtTimes(times, values); }
            catch (e) { for (var i = 0; i < times.length; i++) prop.setValueAtTime(times[i], values[i]); }
            var spatial = prop.propertyValueType === PropertyValueType.TwoD_SPATIAL ||
                          prop.propertyValueType === PropertyValueType.ThreeD_SPATIAL;
            var zero = prop.propertyValueType === PropertyValueType.ThreeD_SPATIAL ? [0, 0, 0] : [0, 0];
            for (var k = 1; k <= prop.numKeys; k++) {
                prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.LINEAR,
                                               KeyframeInterpolationType.LINEAR);
                if (spatial) {
                    prop.setSpatialAutoBezierAtKey(k, false);
                    prop.setSpatialContinuousAtKey(k, false);
                    prop.setSpatialTangentsAtKey(k, zero, zero);
                }
            }
        }
        function mkShape(v, i, o, c) {
            var s = new Shape();
            s.vertices = v; s.inTangents = i; s.outTangents = o; s.closed = c;
            return s;
        }

        STEP = tokganStep(prog, "precomp");
        var pre = app.project.items.addComp("Tokgan Stabilised", data.precomp.width,
                                            data.precomp.height, 1, comp.duration, comp.frameRate);
        pre.motionBlur = true;
        // Same numbering inside, so a frame number reads the same in both.
        tokganDisplayStart(pre, data);

        var nulls = {};
        for (var pi = 0; pi < data.persons.length; pi++) {
            var pd = data.persons[pi];
            STEP = tokganStep(prog, "person " + (pi + 1) + "/" + data.persons.length + " (" + pd.name + ", " +
                                    pd.times.length + " keys)");
            var nl = pre.layers.addNull(pre.duration);
            nl.name = pd.name;
            // Anchor at the Null's own origin, so a child's position is its
            // offset from the pelvis and nothing else.
            nl.transform.anchorPoint.setValue([0, 0]);
            keyLinear(nl.transform.position, scaled(pd.times), pd.pos);
            nulls[pd.pid] = nl;
            pd = null; data.persons[pi] = null;
        }

        for (var s = data.shapes.length - 1; s >= 0; s--) {
            var sd = data.shapes[s];
            STEP = tokganStep(prog, "shape " + (data.shapes.length - s) + "/" + data.shapes.length + " (" +
                                    sd.name + ", " + sd.times.length + " keys)");
            var sl = pre.layers.addShape();
            sl.name = sd.name;
            sl.transform.anchorPoint.setValue([0, 0]);
            // Parent BEFORE keying, without compensation: the keyed values
            // are already in the parent's space.
            if (nulls[sd.pid]) sl.setParentWithJump(nulls[sd.pid]);
            var st = scaled(sd.times);
            keyLinear(sl.transform.position, st, sd.xf.pos);
            keyLinear(sl.transform.rotation, st, sd.xf.rot);
            sl.motionBlur = true;

            var root = sl.property("ADBE Root Vectors Group");
            var grp = root.addProperty("ADBE Vector Group");
            grp.name = sd.name;
            var inside = grp.property("ADBE Vectors Group");
            // AE 25.6: add every property first, then re-acquire references.
            inside.addProperty("ADBE Vector Shape - Group");
            inside.addProperty("ADBE Vector Graphic - Fill");
            var pathItem = null, fillItem = null;
            for (var pp = 1; pp <= inside.numProperties; pp++) {
                var mn = inside.property(pp).matchName;
                if (mn === "ADBE Vector Shape - Group") pathItem = inside.property(pp);
                else if (mn === "ADBE Vector Graphic - Fill") fillItem = inside.property(pp);
            }
            if (fillItem && sd.color) fillItem.property("ADBE Vector Fill Color").setValue(sd.color);
            var shapesAt = new Array(sd.times.length);
            for (var i = 0; i < sd.times.length; i++) {
                shapesAt[i] = mkShape(sd.verts[i], sd.inT[i], sd.outT[i], sd.closed);
            }
            var pathProp = pathItem.property("ADBE Vector Shape");
            try { pathProp.setValuesAtTimes(st, shapesAt); }
            catch (e) { for (var j = 0; j < st.length; j++) pathProp.setValueAtTime(st[j], shapesAt[j]); }

            if (sd.opacity.length > 0) {
                var op = grp.property("ADBE Vector Transform Group").property("ADBE Vector Group Opacity");
                tokganHeldKeys(op, sd.opacity, fpsScale);
            }
            // Keyed: release this part's data.
            shapesAt = null; st = null; sd = null; data.shapes[s] = null;
        }

        STEP = tokganStep(prog, "camera (" + data.camera.times.length + " keys)");
        var lay = comp.layers.add(pre);
        lay.name = "Tokgan Shapes";
        // Layer space == comp space, so the Corner Pin points are plate pixels -- placed at (0.5, 0.5)
        // because those are pixel CENTRES and AE's (0, 0) is the first pixel's top-left corner (shapes
        // otherwise render half a pixel up and left; measured through AE 2025's render, 2026-10-17).
        lay.transform.anchorPoint.setValue([0, 0]);
        lay.transform.position.setValue([0.5, 0.5]);
        lay.motionBlur = true;
        var pin = lay.property("ADBE Effect Parade").addProperty("ADBE Corner Pin");
        var ct = scaled(data.camera.times);
        var corners = data.camera.corners;
        var names = ["ADBE Corner Pin-0001", "ADBE Corner Pin-0002",
                     "ADBE Corner Pin-0003", "ADBE Corner Pin-0004"];
        for (var c = 0; c < 4; c++) {
            var vals = new Array(corners.length);
            for (var k2 = 0; k2 < corners.length; k2++) vals[k2] = corners[k2][c];
            keyLinear(pin.property(names[c]), ct, vals);
        }
        vals = null; corners = null; ct = null;

        tokganCommentNote(comp, timing.note);
        app.endUndoGroup();
        var summary = "Imported " + data.persons.length + " person(s), " + data.shapes.length +
                      " part layers, " + data.camera.times.length + " camera keys";
        data = null;
        tokganGc();
        tokganDone(prog, summary, timing.note);
    } catch (err) {
        try { app.endUndoGroup(); } catch (e2) {}
        raw = null; data = null;
        tokganGc();
        tokganFailed(prog, STEP, err);
        // The one modal dialog: a failure, after the undo group is closed.
        alert("Tokgan import failed.\nStep: " + STEP + "\nError: " + err.toString() +
              "\nLine:  " + (err.line || "?"));
    }
})();
"""

assert LOADER_HIER_TEMPLATE.isascii(), "LOADER_HIER_TEMPLATE must be pure ASCII"


# Solid + masks loader. One solid per person, one Add mask per body
# part. Asked for because the shape-layer import is slow on dense clips, and
# solid + masks is what AE riggers expect. Mask Path keys go in with ONE
# setValuesAtTimes per mask and no per-key fallback: the fallback is what made
# the shape path slow when it fired, and the sidecar is validated in Python
# before it is written, so a refusal here is a real error to report.
# Same substitution rules as LOADER_HIER_TEMPLATE.
LOADER_MASKS_TEMPLATE = r"""// Auto-generated AE loader for Tokgan shape data (Solid + masks).
// Sidecar JSON (__DATA_BASENAME__) is expected next to this script.
(function() {
__COMMON__
    var STEP = "init";
    var prog = tokganProgressOpen("Tokgan import (Solid + masks)");
    try {
        var expectedName = "__DATA_BASENAME__";
        STEP = tokganStep(prog, "find-sidecar (" + expectedName + ")");
        var dataFile = tokganFindSidecar(expectedName);
        if (!dataFile) {
            tokganStep(prog, "cancelled: no sidecar");
            return;
        }

        STEP = tokganStep(prog, "read-sidecar (" + dataFile.fsName + ", " + dataFile.length + " bytes)");
        dataFile.encoding = "UTF-8";
        if (!dataFile.open("r")) throw new Error("File.open returned false for " + dataFile.fsName);
        var raw = dataFile.read();
        dataFile.close();

        STEP = tokganStep(prog, "json-parse (" + raw.length + " chars)");
        var t0 = (new Date()).getTime();
        var data = tokganParse(raw);
        raw = null;
        var tParse = ((new Date()).getTime() - t0) / 1000;
        if (data.mode !== "masks" || !data.solids) {
            throw new Error(dataFile.displayName + " is not a Tokgan Solid + masks sidecar; " +
                            "re-run tokgan_json_to_ae.py --masks on the raw JSON.");
        }

        app.beginUndoGroup("Import Tokgan Masks");

        STEP = tokganStep(prog, "init-comp (parsed in " + tParse.toFixed(1) + " s)");
        var comp = app.project.activeItem;
        if (!(comp instanceof CompItem)) {
            comp = app.project.items.addComp("TokganShapes", data.width, data.height, 1,
                                             data.duration, data.fps);
        }
        comp.motionBlur = true;
        tokganDisplayStart(comp, data);
        var timing = tokganTiming(comp, data);
        var tBuild = (new Date()).getTime();

        // Re-runs replace cleanly: every solid this loader made is named
        // "Tokgan Masks ...". (The old solids' footage stays in the Solids
        // folder, as AE leaves it for any deleted solid layer.)
        for (var rl = comp.numLayers; rl >= 1; rl--) {
            if (comp.layer(rl).name.indexOf("Tokgan Masks") === 0) comp.layer(rl).remove();
        }

        function scaled(times) {
            var out = new Array(times.length);
            for (var i = 0; i < times.length; i++) out[i] = times[i] * timing.scale;
            return out;
        }
        function mkShape(v, i, o, c) {
            var s = new Shape();
            s.vertices = v; s.inTangents = i; s.outTangents = o; s.closed = c;
            return s;
        }

        var nMasks = 0;
        // Last person first: addSolid puts each new layer on top, so the
        // first person ends up as the top layer, as in the other layouts.
        for (var p = data.solids.length - 1; p >= 0; p--) {
            var sd = data.solids[p];
            STEP = tokganStep(prog, "solid " + (data.solids.length - p) + "/" + data.solids.length + " (" +
                                    sd.name + ", " + sd.masks.length + " masks)");
            // Comp-sized and at the comp's centre (addSolid's default), so
            // layer space == comp space; the half-pixel centre is already in
            // the vertices.
            var layer = comp.layers.addSolid(sd.color, sd.name, data.width, data.height, 1,
                                             comp.duration);
            layer.motionBlur = true;

            for (var m = 0; m < sd.masks.length; m++) {
                var md = sd.masks[m];
                STEP = tokganStep(prog, "mask " + (m + 1) + "/" + sd.masks.length + " (" + md.name + ", " +
                                        md.times.length + " keys)");
                // AE 25.6 invalidates earlier property references when the
                // parent changes, so re-acquire the new mask by index.
                var parade = layer.property("ADBE Mask Parade");
                parade.addProperty("ADBE Mask Atom");
                var mask = parade.property(parade.numProperties);
                mask.name = md.name;
                mask.maskMode = MaskMode.ADD;
                mask.property("ADBE Mask Feather").setValue([0, 0]);

                var shapesAt = new Array(md.times.length);
                for (var i = 0; i < md.times.length; i++) {
                    shapesAt[i] = mkShape(md.verts[i], md.inT[i], md.outT[i], md.closed);
                }
                mask.property("ADBE Mask Shape").setValuesAtTimes(scaled(md.times), shapesAt);

                // Visibility: Mask Opacity 0/100, held, so a part that is
                // not in a frame adds nothing to the solid's alpha there.
                if (md.opacity.length > 0) {
                    var op = mask.property("ADBE Mask Opacity");
                    var ot = new Array(md.opacity.length), ov = new Array(md.opacity.length);
                    for (var q = 0; q < md.opacity.length; q++) {
                        ot[q] = md.opacity[q][0] * timing.scale;
                        ov[q] = md.opacity[q][1];
                    }
                    op.setValuesAtTimes(ot, ov);
                    for (var k = 1; k <= op.numKeys; k++) {
                        op.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD,
                                                     KeyframeInterpolationType.HOLD);
                    }
                }
                nMasks++;
                // Keyed: release this mask's data.
                shapesAt = null; md = null; sd.masks[m] = null;
            }
            sd = null; data.solids[p] = null;
        }

        tokganCommentNote(comp, timing.note);
        var tDone = ((new Date()).getTime() - tBuild) / 1000;
        app.endUndoGroup();
        var summary = "Imported " + data.solids.length + " solid(s), " + nMasks + " masks (JSON parse " +
                      tParse.toFixed(1) + " s, AE build " + tDone.toFixed(1) + " s)";
        data = null;
        tokganGc();
        tokganDone(prog, summary, timing.note);
    } catch (err) {
        try { app.endUndoGroup(); } catch (e2) {}
        raw = null; data = null;
        tokganGc();
        tokganFailed(prog, STEP, err);
        // The one modal dialog: a failure, after the undo group is closed.
        alert("Tokgan import failed.\nStep: " + STEP + "\nError: " + err.toString() +
              "\nLine:  " + (err.line || "?"));
    }
})();
"""

assert LOADER_MASKS_TEMPLATE.isascii(), "LOADER_MASKS_TEMPLATE must be pure ASCII"


def build_masks_payload(data, keep_names, person_color, width, height, fps,
                        start_frame, end_frame):
    """Payload for the Solid + masks loader: one solid per person (in person
    id order), one mask record per body part with the half-pixel centre
    baked into its vertices (PIXEL_CENTRE). Raises SystemExit when a record
    is malformed, because the loader has no per-key fallback to hide it."""
    by_pid = {}
    for obj_name in keep_names:
        obj = data["objects"][obj_name]
        rec = build_shape_record(obj_name, obj, height, start_frame, end_frame, fps)
        if not rec:
            continue
        rec["verts"] = [[[round(x + PIXEL_CENTRE, 2), round(y + PIXEL_CENTRE, 2)] for x, y in vs]
                        for vs in rec["verts"]]
        for i, vs in enumerate(rec["verts"]):
            if not (len(vs) == len(rec["inT"][i]) == len(rec["outT"][i])) or not vs:
                raise SystemExit(
                    "%s: key %d has %d vertices, %d in- and %d out-tangents"
                    % (obj_name, i + 1, len(vs), len(rec["inT"][i]), len(rec["outT"][i])))
        by_pid.setdefault(person_id_of(obj_name, obj), []).append(rec)
    solids = [{"pid": pid, "name": "Tokgan Masks p%d" % pid,
               "color": person_color[pid], "masks": masks}
              for pid, masks in sorted(by_pid.items())]
    return {
        "schema": PAYLOAD_SCHEMA,
        "mode": "masks",
        "width": width,
        "height": height,
        "fps": fps,
        "duration": (end_frame - start_frame + 1) / fps,
        "solids": solids,
    }


class ConvertResult(NamedTuple):
    jsx: str          # the loader to run in After Effects
    data: str         # its sidecar, which must stay next to it
    mode: str         # "hierarchy", "flat" or "masks"
    shapes: object    # shape/part count, or "?" when a cached sidecar was reused
    held: list        # hierarchy notes for missing/failed tracking that was held


def _frame_start(data):
    """The JSON's `frame_start` (the plate's number for JSON frame 1), or None
    when it is absent or not a whole number >= 0."""
    v = data.get("frame_start")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != int(v):
        return None
    return int(v)


def _clip_frames(data):
    """Every JSON frame key the clip mentions: object frames and visibility,
    plus the camera and persons blocks of a v3 JSON. Their max is the last
    frame of the clip as far as this file can tell; the first is always 1."""
    frames = set()
    for obj in data["objects"].values():
        frames.update(int(k) for k in obj.get("frames", {}))
        frames.update(int(k) for k in obj.get("visibility", {}))
    for block in ("camera", "persons"):
        if isinstance(data.get(block), dict):
            frames.update(int(k) for k in data[block] if str(k).lstrip("-").isdigit())
    return frames


def convert(in_path_arg, out_jsx_arg=None, *, flat=False, layout="shapes",
            keep_background=False, fps_override=None, force=False,
            max_cp=DEFAULT_MAX_CP, log=print):
    """Write the AE loader (.jsx) and its sidecar for one shape JSON.

    `layout="masks"` writes the Solid + masks loader (always flat);
    "shapes" (the default) the shape-layer one, a hierarchy for a v3 JSON
    unless `flat`. Path keys over `max_cp` control points are resampled down
    to it (0 = keep every point).

    The command line is a thin wrapper around this; Rotobot Queue calls it
    directly. `log` receives the same progress lines the CLI prints.
    """
    if layout not in LAYOUTS:
        raise ValueError("layout must be one of %s, not %r" % (", ".join(LAYOUTS), layout))
    requested = "masks" if layout == "masks" else ("flat" if flat else "auto")
    max_cp = int(max_cp or 0)
    if max_cp < 0:
        raise ValueError("max_cp must be 0 (no cap) or more, not %d" % max_cp)

    in_path = in_path_arg
    out_jsx = out_jsx_arg or os.path.splitext(in_path)[0] + ".jsx"
    out_data = os.path.splitext(out_jsx)[0] + "_data.json"

    # Reuse the sidecar only if (a) the user didn't force, (b) it's newer
    # than the input, AND (c) it was written by a code build that produced
    # the same payload schema. The schema check makes adding new fields
    # (like `color`) safe -- older sidecars get rebuilt automatically.
    sidecar_schema_ok = False
    sidecar_mode = "flat"
    if os.path.exists(out_data):
        try:
            with open(out_data, encoding="utf-8") as _sf:
                _side = json.load(_sf)
            # --flat changes the output for the same input, so a sidecar is
            # only reusable when it was built with the same request.
            sidecar_schema_ok = (_side.get("schema") == PAYLOAD_SCHEMA
                                 and _side.get("requested") == requested
                                 and _side.get("max_cp") == max_cp)
            sidecar_mode = _side.get("mode", "flat")
        except (OSError, ValueError):
            sidecar_schema_ok = False

    # --fps and --keep-background change sidecar contents without changing
    # the input file's mtime, so an mtime-only cache check would silently
    # serve a stale sidecar that ignores the flag. Force rebuild whenever
    # either is set.
    data_is_current = (
        not force
        and not keep_background
        and fps_override is None
        and sidecar_schema_ok
        and os.path.getmtime(out_data) >= os.path.getmtime(in_path)
    )

    summary_extra = ""
    mode = sidecar_mode

    if data_is_current:
        log(f"Reusing existing data file (newer than input): {out_data}")
        n_shapes = "?"  # not loaded; reported as ? in the summary
    else:
        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)
        n_capped = cap_control_points(data, max_cp)
        if n_capped:
            log(f"  Control points: {n_capped} path key(s) resampled to {max_cp}")

        res = data.get("resolution", [data.get("width", 2160), data.get("height", 4096)])
        width, height = int(res[0]), int(res[1])
        json_fps = float(data.get("fps", 24))
        fps = float(fps_override) if fps_override else json_fps
        if fps_override and fps_override != json_fps:
            log(f"  fps override: JSON says {json_fps}, using {fps}")
        frame_start = _frame_start(data)
        # Is the fps real? Before frame_start existed the engine stamped every
        # JSON fps=24 whatever the plate ran at, so a plain 24 with no
        # frame_start could be anything, and the loader keeps retiming those
        # to one source frame per comp frame. frame_start, any other fps, or
        # an explicit --fps is a real rate, keyed at its real times.
        retime = frame_start is None and json_fps == 24 and not fps_override

        # Time zero is the clip's first frame, JSON frame 1, for every layer.
        # It used to be min() of the frames with shapes in them, which keyed
        # a person who entered at frame 30 twenty-nine frames early against
        # the plate -- Silhouette and Nuke key by absolute frame and never
        # were. min(1, ...) only matters to a hand-made JSON keyed from 0.
        all_frames = _clip_frames(data)
        start_frame = min([1] + list(all_frames))
        end_frame = max(all_frames) if all_frames else 48
        n_frames = end_frame - start_frame + 1
        duration = n_frames / fps

        person_color, fg_ids, bg_ids = classify_persons(
            data["objects"], width, height
        )
        bg_set = set(bg_ids)

        keep_names = [
            name for name, obj in data["objects"].items()
            if keep_background or person_id_of(name, obj) not in bg_set
        ]
        has_v3 = bool(data.get("camera") or data.get("persons"))
        if layout == "masks":
            mode = "masks"
        else:
            mode = "hierarchy" if (has_v3 and not flat) else "flat"

        if mode == "masks":
            payload = build_masks_payload(data, keep_names, person_color, width,
                                          height, fps, start_frame, end_frame)
            shapes = [m for sd in payload["solids"] for m in sd["masks"]]
            summary_extra += (
                f"  Solid + masks: {len(payload['solids'])} solid(s), "
                f"{len(shapes)} masks\n")
        elif mode == "hierarchy":
            # The hierarchy reads the JSON itself; hand it the capped one.
            payload = build_hierarchy_payload(
                io.StringIO(json.dumps(data)) if n_capped else in_path, data, set(keep_names), person_color,
                width, height, fps, start_frame, end_frame)
            shapes = payload["shapes"]
            pre = payload["precomp"]
            summary_extra += (
                f"  Hierarchy: {len(payload['persons'])} person Null(s), "
                f"{len(shapes)} part layers, camera Corner Pin on "
                f"{len(payload['camera']['times'])} frames; stabilised "
                f"precomp {pre['width']}x{pre['height']}\n"
            )
            if payload["held"]:
                summary_extra += f"  Held (missing/failed data): {len(payload['held'])} frame(s)\n"
        else:
            shapes = []
            for obj_name in keep_names:
                obj = data["objects"][obj_name]
                pid = person_id_of(obj_name, obj)
                rec = build_shape_record(obj_name, obj, height, start_frame, end_frame, fps)
                if rec:
                    rec["color"] = person_color[pid]
                    shapes.append(rec)
            payload = {
                "schema": PAYLOAD_SCHEMA,
                "mode": "flat",
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "shapes": shapes,
            }
        payload["requested"] = requested
        payload["max_cp"] = max_cp
        payload["retime"] = retime
        if frame_start is not None:
            payload["frame_start"] = frame_start

        with open(out_data, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, separators=(",", ":"))

        n_shapes = len(shapes)
        bg_state = "kept" if keep_background else "dropped"
        summary_extra += (
            f"  Persons: {len(fg_ids)} foreground, "
            f"{len(bg_ids)} background ({bg_state})\n"
        )
        if fg_ids:
            summary_extra += "  Hues:\n"
            for pid in fg_ids:
                r, g, b = person_color[pid]
                summary_extra += (
                    f"    p{pid}: ({int(round(r*255))},{int(round(g*255))},{int(round(b*255))})\n"
                )

    if mode == "hierarchy":
        loader = (LOADER_HIER_TEMPLATE.replace("__DATA_BASENAME__", os.path.basename(out_data))
                  .replace("__COMMON__", COMMON_JS))
    elif mode == "masks":
        loader = (LOADER_MASKS_TEMPLATE.replace("__DATA_BASENAME__", os.path.basename(out_data))
                  .replace("__COMMON__", COMMON_JS))
    else:
        loader = LOADER_TEMPLATE.format(data_basename=os.path.basename(out_data),
                                        common=COMMON_JS)
    with open(out_jsx, "w", encoding="utf-8", newline="\n") as f:
        f.write(loader)

    result = ConvertResult(out_jsx, out_data, mode, n_shapes,
                           list(payload.get("held", [])) if not data_is_current else [])
    log(f"Wrote {out_jsx}  ({os.path.getsize(out_jsx):,} bytes)")
    log(f"Data {out_data}  ({os.path.getsize(out_data):,} bytes)")
    log(f"  Shapes: {n_shapes}")
    if summary_extra:
        log(summary_extra.rstrip("\n"))
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(
        prog="tokgan_json_to_ae",
        description=(
            "Convert a Tokgan JSON shape file into an After Effects loader "
            "(.jsx) plus a compact sidecar data JSON. Run the resulting "
            ".jsx in AE via File > Scripts > Run Script File... A v2 JSON "
            "builds one shape layer holding an animated Bezier path per "
            "object; a v3 JSON (camera + person data) builds a camera Corner "
            "Pin, a Null per person and a shape layer per body part. --masks "
            "builds one solid per person with a mask per body part instead."
        ),
        epilog=(
            "Notes: comp time 0 is the clip's first frame. A comp the loader "
            "creates runs at the JSON's fps; when the JSON has frame_start, "
            "the comp's timeline is numbered from it. A JSON without "
            "frame_start that says 24 fps (every file before v0.20) is "
            "retimed to one source frame per comp frame. Enable 'Allow "
            "Scripts to Write Files "
            "and Access Network' in AE's preferences (Scripting & "
            "Expressions) before running the loader."
        ),
    )
    p.add_argument("input", help="Path to input Tokgan JSON file")
    p.add_argument(
        "output",
        nargs="?",
        help=(
            "Path to write the AE loader .jsx (default: same dir/name as "
            "the input with .jsx extension). The sidecar will be written "
            "next to it as <output_stem>_data.json."
        ),
    )
    p.add_argument(
        "-f", "--force",
        action="store_true",
        help="Rebuild the sidecar data file even if it's newer than the input.",
    )
    p.add_argument(
        "--keep-background",
        action="store_true",
        help=(
            "Import 'background' persons (max-over-time bbox dim < "
            "max(comp_w, comp_h)/%d) as a single neutral-grey fill "
            "instead of dropping them. Off by default to save AE "
            "keyframe-writing time on crowded clips." % BACKGROUND_DIM_DIVISOR
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Override the fps declared in the JSON metadata. Tokgan "
            "exports stamp every file with fps=24 even when the source "
            "video runs at e.g. 25 fps, which makes the shapes play "
            "~4%% faster than the matching footage in AE. Pass the "
            "footage's true fps (e.g. --fps 25) to fix the timing."
        ),
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Always build the single 'Tokgan Shapes' layer, even for a v3 "
            "JSON with camera/person data (which otherwise builds the "
            "camera / person / body-part hierarchy)."
        ),
    )
    p.add_argument(
        "--masks",
        action="store_true",
        help=(
            "Solid + masks: one solid per person and one Add mask per body "
            "part, its Mask Path keyed per frame and its Mask Opacity held "
            "0/100 for visibility. Faster to import than shape layers on "
            "dense clips. Always flat (no camera / person hierarchy)."
        ),
    )
    p.add_argument(
        "--max-cp",
        type=int,
        default=DEFAULT_MAX_CP,
        metavar="N",
        help=(
            "Resample any path key with more than N control points to N, "
            "spaced evenly along the curve (default %d; 0 keeps every "
            "point). Every point is keyed on every frame, so dense shapes "
            "make long shots slow to import." % DEFAULT_MAX_CP
        ),
    )
    args = p.parse_args()
    if args.max_cp < 0:
        p.error("--max-cp must be 0 (no cap) or more")
    convert(args.input, args.output, flat=args.flat,
            layout="masks" if args.masks else "shapes",
            keep_background=args.keep_background, fps_override=args.fps,
            force=args.force, max_cp=args.max_cp)


if __name__ == "__main__":
    main()
