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
exists in. The loader retimes all keys to the active comp's frame rate so
each source frame lands on one comp frame regardless of the JSON's declared
fps. Motion blur is enabled on the layer.

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

Use `python3 tokgan_json_to_ae.py --help` for invocation details.

Coordinate convention:
    Both the JSON and AE use Y-down, origin top-left pixel coordinates.
    Bezier handles in JSON are absolute (left_x/left_y, right_x/right_y);
    AE expects them as deltas from the vertex, so we subtract.
"""

import colorsys
import json
import os
import sys


# Persons whose maximum-over-time bounding-box max dimension is smaller
# than max(comp_w, comp_h) / BACKGROUND_DIM_DIVISOR are treated as
# background figures: by default dropped from the import; with
# --keep-background they are imported but share one neutral-grey fill.
BACKGROUND_DIM_DIVISOR = 20

# Bump when the sidecar payload schema changes in a way the loader
# would have to handle. Sidecars whose stored value differs from this
# constant are rebuilt on the next run regardless of mtime, so older
# sidecars without `color` don't quietly stay around.
PAYLOAD_SCHEMA = 3

# Hierarchy mode: the stabilised precomp is padded this far beyond the
# stabilised extent of every vertex, and must stay within AE's comp limit.
PRECOMP_MARGIN = 64
AE_MAX_COMP_SIZE = 30000

BACKGROUND_GREY = [0.5, 0.5, 0.5]


def round_pt(x, y, ndigits=2):
    return [round(x, ndigits), round(y, ndigits)]


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
    # iff f has path data (fallback when visibility is absent). Each visible
    # segment emits four hold-stepped keys so AE shows a clean on/off step.
    visibility = obj.get("visibility", {})
    if visibility:
        exists = sorted(int(k) for k, v in visibility.items() if v)
    else:
        exists = list(path_frames)

    opacity = []
    if exists:
        # Build contiguous segments
        segments = []
        s = exists[0]; e = exists[0]
        for f in exists[1:]:
            if f == e + 1:
                e = f
            else:
                segments.append((s, e))
                s, e = f, f
        segments.append((s, e))

        def _t(frame):
            return round((frame - start_frame) / fps, 6)

        raw = []
        for seg_start, seg_end in segments:
            if seg_start > start_frame:
                raw.append((_t(seg_start - 1), 0))
            raw.append((_t(seg_start), 100))
            raw.append((_t(seg_end), 100))
            if seg_end < end_frame:
                raw.append((_t(seg_end + 1), 0))
        if segments[0][0] > start_frame and not any(t == 0.0 for t, _ in raw):
            raw.append((0.0, 0))

        seen = {}
        for t, v in raw:
            seen[t] = v
        opacity = [[t, v] for t, v in sorted(seen.items())]

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


LOADER_TEMPLATE = r"""// Auto-generated AE loader for Tokgan shape data.
// Sidecar JSON ({data_basename}) is expected next to this script.
(function() {{
    var STEP = "init";
    try {{
        STEP = "locate-script";
        var scriptFile = new File($.fileName);
        var parentDir = scriptFile.parent;
        var expectedName = "{data_basename}";

        STEP = "find-sidecar";
        // Use getFiles() to enumerate the directory rather than constructing
        // a File from a path string -- robust to ':' / spaces in basenames.
        var siblings = parentDir.getFiles();
        var dataFile = null;
        for (var fi = 0; fi < siblings.length; fi++) {{
            var f = siblings[fi];
            if (f instanceof File && f.displayName === expectedName) {{
                dataFile = f;
                break;
            }}
        }}
        if (!dataFile) {{
            // Fallback: ask the user to pick it.
            // The .jsx alone is useless: the sidecar must travel with it.
            dataFile = File.openDialog(
                "Locate sidecar: " + expectedName +
                " (generated by tokgan_json_to_ae.py, NOT the raw Tokgan JSON)"
            );
            if (!dataFile) return;
        }}

        STEP = "read-sidecar (" + dataFile.fsName + ", " + dataFile.length + " bytes)";
        dataFile.encoding = "UTF-8";
        if (!dataFile.open("r")) {{
            throw new Error("File.open returned false for " + dataFile.fsName);
        }}
        var raw = dataFile.read();
        dataFile.close();
        if (!raw || raw.length === 0) {{
            throw new Error("File.read returned empty string");
        }}

        STEP = "json-parse (" + raw.length + " chars)";
        var t0 = (new Date()).getTime();
        var data;
        if (typeof JSON !== "undefined" && JSON.parse) {{
            data = JSON.parse(raw);
        }} else {{
            // Polyfill: JSON is valid JS; eval inside parens.
            // Safe here because the data is produced by our own Python.
            data = eval("(" + raw + ")");
        }}
        var tParse = ((new Date()).getTime() - t0) / 1000;
        if (!data.shapes || data.duration === undefined) {{
            throw new Error(
                dataFile.displayName + " is not a Tokgan sidecar (no shapes/duration). " +
                "If it is the raw Tokgan JSON, run tokgan_json_to_ae.py on it and " +
                "keep the generated " + expectedName + " next to this .jsx."
            );
        }}

        STEP = "init-comp";
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

        STEP = "create-shape-layer";
        // Remove a previous "Tokgan Shapes" layer so re-runs replace cleanly.
        for (var rl = comp.numLayers; rl >= 1; rl--) {{
            if (comp.layer(rl).name === "Tokgan Shapes") comp.layer(rl).remove();
        }}
        // Single shape layer holding every imported path as a sub-group.
        var layer = comp.layers.addShape();
        layer.name = "Tokgan Shapes";
        var root = layer.property("ADBE Root Vectors Group");

        // Re-time keys to the *comp's* frame rate so each source frame
        // lands on one comp frame, regardless of the fps the JSON declared.
        // (Common when the JSON metadata fps disagrees with the comp.)
        var fpsScale = data.fps / comp.frameRate;
        $.writeln("[Tokgan] data fps=" + data.fps + " comp fps=" + comp.frameRate + " scale=" + fpsScale);

        STEP = "build-groups";
        var shapes = data.shapes;
        for (var s = 0; s < shapes.length; s++) {{
            var sd = shapes[s];
            STEP = "group " + (s + 1) + "/" + shapes.length + " (" + sd.name + ")";

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
                for (var j = 0; j < sd.opacity.length; j++) {{
                    op.setValueAtTime(sd.opacity[j][0] * fpsScale, sd.opacity[j][1]);
                    op.setInterpolationTypeAtKey(
                        op.numKeys,
                        KeyframeInterpolationType.HOLD,
                        KeyframeInterpolationType.HOLD
                    );
                }}
            }}
        }}

        STEP = "layer-transform-and-motion-blur";
        // Put the layer's origin at comp (0,0) so absolute pixel coords in
        // the path data line up with comp coords (no half-resolution shift).
        layer.transform.anchorPoint.setValue([0, 0]);
        layer.transform.position.setValue([0, 0]);
        layer.motionBlur = true;

        var tDone = ((new Date()).getTime() - tBuild) / 1000;
        app.endUndoGroup();

        $.writeln(
            "[Tokgan import] " + shapes.length + " layers; " +
            "JSON parse " + tParse.toFixed(2) + "s; " +
            "AE build " + tDone.toFixed(2) + "s"
        );
    }} catch (err) {{
        try {{ app.endUndoGroup(); }} catch (e2) {{}}
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
    var STEP = "init";
    try {
        STEP = "find-sidecar";
        var expectedName = "__DATA_BASENAME__";
        var siblings = (new File($.fileName)).parent.getFiles();
        var dataFile = null;
        for (var fi = 0; fi < siblings.length; fi++) {
            if (siblings[fi] instanceof File && siblings[fi].displayName === expectedName) {
                dataFile = siblings[fi];
                break;
            }
        }
        if (!dataFile) {
            dataFile = File.openDialog(
                "Locate sidecar: " + expectedName +
                " (generated by tokgan_json_to_ae.py, NOT the raw Tokgan JSON)");
            if (!dataFile) return;
        }

        STEP = "read-sidecar (" + dataFile.fsName + ")";
        dataFile.encoding = "UTF-8";
        if (!dataFile.open("r")) throw new Error("File.open returned false for " + dataFile.fsName);
        var raw = dataFile.read();
        dataFile.close();

        STEP = "json-parse (" + raw.length + " chars)";
        var data = (typeof JSON !== "undefined" && JSON.parse) ? JSON.parse(raw) : eval("(" + raw + ")");
        if (data.mode !== "hierarchy" || !data.camera || !data.precomp) {
            throw new Error(dataFile.displayName + " is not a Tokgan hierarchy sidecar; " +
                            "re-run tokgan_json_to_ae.py on the raw JSON.");
        }

        app.beginUndoGroup("Import Tokgan Hierarchy");

        STEP = "init-comp";
        var comp = app.project.activeItem;
        if (!(comp instanceof CompItem)) {
            comp = app.project.items.addComp("TokganShapes", data.width, data.height, 1,
                                             data.duration, data.fps);
        }
        comp.motionBlur = true;
        var fpsScale = data.fps / comp.frameRate;

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

        STEP = "precomp";
        var pre = app.project.items.addComp("Tokgan Stabilised", data.precomp.width,
                                            data.precomp.height, 1, comp.duration, comp.frameRate);
        pre.motionBlur = true;

        STEP = "persons";
        var nulls = {};
        for (var pi = 0; pi < data.persons.length; pi++) {
            var pd = data.persons[pi];
            var nl = pre.layers.addNull(pre.duration);
            nl.name = pd.name;
            // Anchor at the Null's own origin, so a child's position is its
            // offset from the pelvis and nothing else.
            nl.transform.anchorPoint.setValue([0, 0]);
            keyLinear(nl.transform.position, scaled(pd.times), pd.pos);
            nulls[pd.pid] = nl;
        }

        STEP = "shapes";
        for (var s = data.shapes.length - 1; s >= 0; s--) {
            var sd = data.shapes[s];
            STEP = "shape " + (s + 1) + "/" + data.shapes.length + " (" + sd.name + ")";
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
                for (var q = 0; q < sd.opacity.length; q++) {
                    op.setValueAtTime(sd.opacity[q][0] * fpsScale, sd.opacity[q][1]);
                    op.setInterpolationTypeAtKey(op.numKeys, KeyframeInterpolationType.HOLD,
                                                 KeyframeInterpolationType.HOLD);
                }
            }
        }

        STEP = "camera";
        var lay = comp.layers.add(pre);
        lay.name = "Tokgan Shapes";
        // Layer space == comp space, so the Corner Pin points are plate pixels.
        lay.transform.anchorPoint.setValue([0, 0]);
        lay.transform.position.setValue([0, 0]);
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

        app.endUndoGroup();
        $.writeln("[Tokgan hierarchy] " + data.persons.length + " persons, " +
                  data.shapes.length + " parts, " + data.camera.times.length + " camera keys");
    } catch (err) {
        try { app.endUndoGroup(); } catch (e2) {}
        alert("Tokgan import failed.\nStep: " + STEP + "\nError: " + err.toString() +
              "\nLine:  " + (err.line || "?"));
    }
})();
"""

assert LOADER_HIER_TEMPLATE.isascii(), "LOADER_HIER_TEMPLATE must be pure ASCII"


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
            "Pin, a Null per person and a shape layer per body part."
        ),
        epilog=(
            "Notes: each source frame lands on one comp frame regardless of "
            "the JSON's declared fps. Enable 'Allow Scripts to Write Files "
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
    args = p.parse_args()
    requested = "flat" if args.flat else "auto"

    in_path = args.input
    out_jsx = args.output or os.path.splitext(in_path)[0] + ".jsx"
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
                                 and _side.get("requested") == requested)
            sidecar_mode = _side.get("mode", "flat")
        except (OSError, ValueError):
            sidecar_schema_ok = False

    # --fps and --keep-background change sidecar contents without changing
    # the input file's mtime, so an mtime-only cache check would silently
    # serve a stale sidecar that ignores the flag. Force rebuild whenever
    # either is set.
    data_is_current = (
        not args.force
        and not args.keep_background
        and args.fps is None
        and sidecar_schema_ok
        and os.path.getmtime(out_data) >= os.path.getmtime(in_path)
    )

    summary_extra = ""
    mode = sidecar_mode

    if data_is_current:
        print(f"Reusing existing data file (newer than input): {out_data}")
        n_shapes = "?"  # not loaded; reported as ? in the summary
    else:
        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        res = data.get("resolution", [data.get("width", 2160), data.get("height", 4096)])
        width, height = int(res[0]), int(res[1])
        json_fps = float(data.get("fps", 24))
        fps = float(args.fps) if args.fps else json_fps
        if args.fps and args.fps != json_fps:
            print(f"  fps override: JSON says {json_fps}, using {fps}")

        all_frames = set()
        for obj in data["objects"].values():
            for fkey in obj.get("frames", {}).keys():
                all_frames.add(int(fkey))
        start_frame = min(all_frames) if all_frames else 1
        end_frame = max(all_frames) if all_frames else 48
        n_frames = end_frame - start_frame + 1
        duration = n_frames / fps

        person_color, fg_ids, bg_ids = classify_persons(
            data["objects"], width, height
        )
        bg_set = set(bg_ids)

        keep_names = [
            name for name, obj in data["objects"].items()
            if args.keep_background or person_id_of(name, obj) not in bg_set
        ]
        has_v3 = bool(data.get("camera") or data.get("persons"))
        mode = "hierarchy" if (has_v3 and not args.flat) else "flat"

        if mode == "hierarchy":
            payload = build_hierarchy_payload(
                in_path, data, set(keep_names), person_color,
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

        with open(out_data, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, separators=(",", ":"))

        n_shapes = len(shapes)
        bg_state = "kept" if args.keep_background else "dropped"
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
        loader = LOADER_HIER_TEMPLATE.replace("__DATA_BASENAME__", os.path.basename(out_data))
    else:
        loader = LOADER_TEMPLATE.format(data_basename=os.path.basename(out_data))
    with open(out_jsx, "w", encoding="utf-8", newline="\n") as f:
        f.write(loader)

    print(f"Wrote {out_jsx}  ({os.path.getsize(out_jsx):,} bytes)")
    print(f"Data {out_data}  ({os.path.getsize(out_data):,} bytes)")
    print(f"  Shapes: {n_shapes}")
    if summary_extra:
        print(summary_extra, end="")


if __name__ == "__main__":
    main()
