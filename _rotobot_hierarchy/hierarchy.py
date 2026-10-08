"""Host-agnostic camera / person / body-part decomposition of a v3 document.

No ``nuke`` import: the After Effects and Silhouette writers use this too.

A v3 ``lozenge_bezier_anim`` document carries, per frame:

* ``camera[f].H2d`` -- a cumulative 3x3 homography (row-major, Y-down plate
  pixels) mapping a pixel of the SEED frame (the first tracked frame) to
  where it is in frame ``f``. It is projective: the ECC tracker solves
  ``MOTION_HOMOGRAPHY``, so it is NOT reduced to an affine here.
* ``persons[f][pid].pelvis_px`` -- the hip midpoint in plate pixels.
* ``objects[key].frames[f].bone`` -- the part's bone, pt0 -> pt1.

:func:`decompose` splits every knot's plate position into three nested
transforms, the hierarchy a compositor animates::

    plate = T1( T2( T3( local ) ) )

    T1  camera      the exact homography H2d(f)            (one per clip)
    T2  person      translate to the pelvis, in stabilised  (one per person)
                    (seed-frame) pixels
    T3  body part   translate to the bone's pt0, relative   (one per object)
                    to the pelvis, then rotate by the bone
                    angle
    local           the knot in bone-local pixels

All coordinates stay Y-down pixels; a writer converts to its host's space.
Angles are ``atan2(dy, dx)`` in degrees in that Y-down space, i.e. positive
is clockwise on screen, and are unwrapped so consecutive frames never jump
by more than 180 degrees.

Missing or bad data is HELD, never replaced by identity or the origin --
holding keeps the composed plate position right, where identity would
teleport the shapes for a frame:

* a camera frame that is absent, ``ecc-failed``, or an ``identity`` entry
  after the track has started (FitEngine writes that when a sidecar is
  missing) holds the last good homography;
* a person frame that is absent or has ``pelvis_px == [0, 0]`` (both hips
  missing) holds the last good pelvis;
* an object frame without a bone holds the last good bone.

The leading frames before the first good value take the first good value.
Every hold is listed in :attr:`Hierarchy.held`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .reader import LozengeDoc, LozengePoint

Mat3 = Tuple[float, float, float, float, float, float, float, float, float]
Vec2 = Tuple[float, float]

IDENTITY: Mat3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

#: Camera ``source`` values that are a real solve. ``""`` is a sidecar that did
#: not say. ``identity`` is only good while nothing better has been seen yet:
#: before the track starts it IS the camera (no tracker ran); after, it is a
#: frame whose sidecar was missing.
_GOOD_SOURCES = {"ecc", "ecc-seed", "sidecar", ""}

#: Tolerance the writers hold a hierarchy to before they will emit it.
ROUND_TRIP_TOLERANCE_PX = 0.05


# --------------------------------------------------------------------- matrix


def mat3_mul(a: Sequence[float], b: Sequence[float]) -> Mat3:
    return tuple(  # type: ignore[return-value]
        sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
        for r in range(3)
        for c in range(3)
    )


def mat3_inv(m: Sequence[float]) -> Mat3:
    a, b, c, d, e, f, g, h, i = m
    co = (
        e * i - f * h, c * h - b * i, b * f - c * e,
        f * g - d * i, a * i - c * g, c * d - a * f,
        d * h - e * g, b * g - a * h, a * e - b * d,
    )
    det = a * co[0] + b * co[3] + c * co[6]
    if abs(det) < 1e-12:
        raise ValueError("singular homography")
    return tuple(v / det for v in co)  # type: ignore[return-value]


def normalise(m: Sequence[float]) -> Mat3:
    """Scale so m[8] == 1. ECC's accumulated H drifts off 1 (0.9956 by
    frame 24 on the v0.10.0 clip); hosts that take corner points do not
    care, but anything reading the matrix entries does."""
    s = m[8]
    if abs(s) < 1e-12:
        raise ValueError("homography with zero scale term")
    return tuple(v / s for v in m)  # type: ignore[return-value]


def apply_h(m: Sequence[float], x: float, y: float) -> Vec2:
    w = m[6] * x + m[7] * y + m[8]
    return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)


def rotate(x: float, y: float, deg: float) -> Vec2:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (c * x - s * y, s * x + c * y)


def unwrap_degrees(angles: Iterable[float]) -> List[float]:
    out: List[float] = []
    for a in angles:
        if out:
            prev = out[-1]
            a = a + 360.0 * round((prev - a) / 360.0)
        out.append(a)
    return out


def corner_pin(m: Sequence[float], width: float, height: float) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    """Where the plate's corners go under ``m``: upper-left, upper-right,
    lower-right, lower-left. A four-corner pin is a projective map, so these
    four points reproduce ``m`` exactly in a host that offers one."""
    return (
        apply_h(m, 0.0, 0.0),
        apply_h(m, width, 0.0),
        apply_h(m, width, height),
        apply_h(m, 0.0, height),
    )


# ------------------------------------------------------------------ hierarchy


@dataclass
class PartXform:
    """T3 for one object on one frame, relative to its person's pelvis."""

    tx: float
    ty: float
    angle: float  # degrees, Y-down, unwrapped


@dataclass
class Hierarchy:
    width: int
    height: int
    #: Every frame from the first to the last object frame. T1 and T2 are
    #: keyed on all of them so a host never interpolates a value the knots
    #: were not computed with.
    frames: List[int]
    #: T1: frame -> normalised H2d (seed -> frame) and its inverse.
    camera: Dict[int, Mat3]
    camera_inv: Dict[int, Mat3]
    #: T2: pid -> frame -> pelvis in stabilised (seed-frame) pixels.
    pelvis: Dict[int, Dict[int, Vec2]]
    #: T3: object key -> frame -> PartXform (only the object's own frames).
    parts: Dict[str, Dict[int, PartXform]]
    #: object key -> frame -> knots in bone-local pixels (handles included).
    knots: Dict[str, Dict[int, List[LozengePoint]]]
    #: object key -> pid
    person_of: Dict[str, int]
    #: Human-readable notes, one per held frame.
    held: List[str] = field(default_factory=list)

    @property
    def pids(self) -> List[int]:
        return sorted(self.pelvis)

    def to_plate(self, key: str, frame: int, x: float, y: float) -> Vec2:
        """Compose T1(T2(T3(x, y))) -- what a host renders for a local point."""
        pid = self.person_of[key]
        part = self.parts[key][frame]
        px, py = self.pelvis[pid][frame]
        rx, ry = rotate(x, y, part.angle)
        return apply_h(self.camera[frame], px + part.tx + rx, py + part.ty + ry)


def _held_series(frames: Sequence[int], good: Dict[int, object], label: str,
                 held: List[str]) -> Dict[int, object]:
    """Fill every frame from the nearest PREVIOUS good value (the first good
    value for leading frames). Notes each filled frame in ``held``."""
    out: Dict[int, object] = {}
    if not good:
        return out
    first = good[min(good)]
    last = None
    for f in frames:
        if f in good:
            last = good[f]
            out[f] = last
        else:
            out[f] = first if last is None else last
            held.append(f"{label}: frame {f} held")
    return out


def _camera_series(doc: LozengeDoc, frames: Sequence[int], held: List[str]) -> Dict[int, Mat3]:
    cams = doc.camera or {}
    good: Dict[int, object] = {}
    started = False
    for f in sorted(cams):
        cf = cams[f]
        if cf.H2d is None:
            continue
        src = cf.source or ""
        if src in _GOOD_SOURCES or (src == "identity" and not started):
            good[f] = normalise(cf.H2d)
            if src != "identity":
                started = True
        # ecc-failed, or identity after the track started: not a solve.
    if not good:
        return {f: IDENTITY for f in frames}
    return _held_series(frames, good, "camera", held)  # type: ignore[return-value]


def _pelvis_series(doc: LozengeDoc, pids: Iterable[int], frames: Sequence[int],
                   held: List[str]) -> Dict[int, Dict[int, Vec2]]:
    persons = doc.persons or {}
    out: Dict[int, Dict[int, Vec2]] = {}
    for pid in pids:
        good: Dict[int, object] = {}
        for f, by_pid in persons.items():
            pf = by_pid.get(pid)
            if pf is None:
                continue
            if pf.pelvis_px[0] == 0.0 and pf.pelvis_px[1] == 0.0:
                continue  # both hips missing
            good[f] = pf.pelvis_px
        if good:
            out[pid] = _held_series(frames, good, f"person {pid}", held)  # type: ignore[assignment]
    return out


def decompose(doc: LozengeDoc) -> Hierarchy:
    w, h = int(doc.resolution[0]), int(doc.resolution[1])
    all_frames = sorted({f for o in doc.objects.values() for f in o.frames})
    frames = list(range(all_frames[0], all_frames[-1] + 1)) if all_frames else []
    held: List[str] = []

    camera = _camera_series(doc, frames, held)
    camera_inv = {f: mat3_inv(m) for f, m in camera.items()}

    person_of = {k: o.person_id for k, o in doc.objects.items()}
    pelvis_plate = _pelvis_series(doc, sorted(set(person_of.values())), frames, held)
    # Stabilise the pelvis: it is measured on the plate, T2 lives under T1.
    pelvis: Dict[int, Dict[int, Vec2]] = {
        pid: {f: apply_h(camera_inv[f], *xy) for f, xy in series.items()}
        for pid, series in pelvis_plate.items()
    }
    for pid in set(person_of.values()) - set(pelvis):
        # No usable pelvis on any frame: the person root sits at the origin
        # and the parts carry the motion. Composition stays exact.
        pelvis[pid] = {f: (0.0, 0.0) for f in frames}
        held.append(f"person {pid}: no pelvis on any frame, root at origin")

    parts: Dict[str, Dict[int, PartXform]] = {}
    knots: Dict[str, Dict[int, List[LozengePoint]]] = {}
    for key, obj in doc.objects.items():
        pid = obj.person_id
        ofr = sorted(obj.frames)
        bones = {f: obj.frames[f].bone for f in ofr if obj.frames[f].bone is not None}
        bone_series = _held_series(ofr, bones, f"{key} bone", held) if bones else {}

        raw: List[Tuple[int, float, float, float]] = []
        for f in ofr:
            inv = camera_inv[f]
            px, py = pelvis[pid][f]
            if bone_series:
                (b0x, b0y), (b1x, b1y) = bone_series[f]  # type: ignore[misc]
                s0 = apply_h(inv, b0x, b0y)
                s1 = apply_h(inv, b1x, b1y)
                ang = math.degrees(math.atan2(s1[1] - s0[1], s1[0] - s0[0]))
                raw.append((f, s0[0] - px, s0[1] - py, ang))
            else:
                raw.append((f, 0.0, 0.0, 0.0))
        unwrapped = unwrap_degrees(a for _, _, _, a in raw)
        parts[key] = {f: PartXform(tx, ty, a) for (f, tx, ty, _), a in zip(raw, unwrapped)}

        kf: Dict[int, List[LozengePoint]] = {}
        for f in ofr:
            inv = camera_inv[f]
            px, py = pelvis[pid][f]
            part = parts[key][f]

            def local(x: float, y: float) -> Vec2:
                sx, sy = apply_h(inv, x, y)
                return rotate(sx - px - part.tx, sy - py - part.ty, -part.angle)

            pts = []
            for p in obj.frames[f].points:
                x, y = local(p.x, p.y)
                lx, ly = local(p.left_x, p.left_y)
                rx, ry = local(p.right_x, p.right_y)
                pts.append(LozengePoint(x, y, lx, ly, rx, ry))
            kf[f] = pts
        knots[key] = kf

    return Hierarchy(w, h, frames, camera, camera_inv, pelvis, parts, knots, person_of, held)


def round_trip_error(doc: LozengeDoc, hier: Hierarchy) -> float:
    """Largest plate-pixel distance between a knot (or handle) in ``doc`` and
    its re-composition through ``hier``. A writer emits the hierarchy only
    when this is within :data:`ROUND_TRIP_TOLERANCE_PX`."""
    worst = 0.0
    for key, obj in doc.objects.items():
        for f, frame in obj.frames.items():
            for p, q in zip(frame.points, hier.knots[key][f]):
                for (ax, ay), (lx, ly) in (
                    ((p.x, p.y), (q.x, q.y)),
                    ((p.left_x, p.left_y), (q.left_x, q.left_y)),
                    ((p.right_x, p.right_y), (q.right_x, q.right_y)),
                ):
                    bx, by = hier.to_plate(key, f, lx, ly)
                    worst = max(worst, math.hypot(ax - bx, ay - by))
    return worst


__all__ = [
    "Hierarchy",
    "IDENTITY",
    "PartXform",
    "ROUND_TRIP_TOLERANCE_PX",
    "apply_h",
    "corner_pin",
    "decompose",
    "mat3_inv",
    "mat3_mul",
    "normalise",
    "rotate",
    "round_trip_error",
    "unwrap_degrees",
]
