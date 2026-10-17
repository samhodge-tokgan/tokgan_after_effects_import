"""Keyframe undersampling for ``LozengeDoc``.

Many captures produce one JSON frame per video frame — hundreds or
thousands of near-identical control-point sets per shape. Nuke stores
every one as a keyframe, making the resulting Roto node slow to
scrub, slow to render and inconvenient to hand-tweak.

This module reduces the per-object temporal keyframe count by running
Ramer-Douglas-Peucker (RDP) on each object's time-series of state
vectors. A state vector encodes **where the segment is in the plate**
(bone origin) and **how its points sit relative to that bone** (the
local control-point layout, after transforming into bone-local
coordinates). Frames whose state vector is a close linear
interpolation of its neighbours are discarded; the remaining
keyframes still describe the full motion to within ``tolerance``.

"A close linear interpolation of its neighbours" means linear **in
time**, which is what a host reconstructs between two keyframes. The
RDP pass measures exactly that. It previously measured perpendicular
distance to the chord in state space, which ignores the time
parameterisation, and so could not tell motion that retraces its own
path apart from motion that had stopped -- see :func:`rdp_reduction`.

The algorithm is a direct port of the ``key_reduction`` branch of
``tokgan_silhouette_import``
(https://github.com/samhodge-aiml/tokgan_silhouette_import/tree/key_reduction),
MIT-licensed, re-shaped to operate on :class:`LozengeDoc` in memory
rather than file-to-file.

Two call shapes are supported on :func:`undersample_doc`:

    Legacy single-pass (PR #1 behaviour) — one bone-local RDP per object
    governed by a single ``tolerance`` kwarg. Useful for v2 JSONs and for
    quick sanity checks.

    Hierarchical composed-tolerance (issue #279) — three sequential RDP
    passes:

      1. camera reference frame over :attr:`LozengeDoc.camera`
      2. person root pose over :attr:`LozengeDoc.persons`
      3. articulation (bone-local with per-frame pelvis subtracted) per
         object

    Each pass gates its own tolerance. The final kept frames per object
    are the **union** of the three passes' retained frames, intersected
    with the object's own frame set — so a frame is kept whenever the
    camera, the person root, or the articulation changed enough to
    warrant a keyframe, and dropped only when all three agree the frame
    is linearly interpolable.

Requirements:
    Each object must carry per-frame ``bone`` endpoints (schema v2 writer
    output). Objects without ``bone`` are left untouched — RDP needs an
    anchor so the state vector tracks "the whole segment moved", not
    "every point jittered by itself".
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

from .reader import (
    CameraFrame,
    LozengeDoc,
    LozengeFrame,
    LozengeObject,
    LozengePoint,
    PersonFrame,
)

# Tolerance presets, picked from a wedge sweep across four real UHD/HD
# Rotobot-Next outputs (1.1 MB → 57.9 MB; 126 → 8198 keyframes). Units
# are mixed pixel + degree — the state vector norm is dominated by the
# bone-origin pixel coordinates on typical plates, so a tolerance of
# ``n`` is roughly "keep a frame if any aspect of its state vector
# diverged by n pixel-equivalents from a linear interpolation of its
# retained neighbours."
TOLERANCE_CONSERVATIVE = 2.0   # ≥99% keyframes kept on measured plates
TOLERANCE_BALANCED = 5.0       # 87–99% kept; the useful mid-ground
TOLERANCE_AGGRESSIVE = 10.0    # ~75% kept; faster scrubs, keyframes
                               # still coarsely preserve articulation
TOLERANCE_VERY_AGGRESSIVE = 25.0  # 45–80% kept; artist review needed

DEFAULT_TOLERANCE = TOLERANCE_BALANCED


# Composed-tolerance presets for the hierarchical path (issue #279).
# `camera_tolerance` + `person_tolerance` are in **screen pixels** (what
# the camera H2d + person pelvis_px are measured in; these encode where
# on the plate to draw the roto). `articulation_tolerance` is a
# **dimensionless fraction of the person's max tapered-capsule radius**
# (body-local, depth-invariant per person — see PR #5).
#
# Values retuned from a 4-clip cross-clip sweep on real UHD footage
# (see benchmarks/real_results_cross_clip.md). Measured retention:
#
# NOTE: the retention figures below PREDATE the time-axis and rotation-unit
# fixes to the articulation metric, and will have moved -- the metric they
# were measured against no longer exists. They are kept because they are
# still the record of how the triples were chosen and of how tightly the
# three clips agreed, but the sweep needs re-running on the same four plates
# before anyone treats them as current. The ORDERING is sound either way
# (fine keeps more than balanced keeps more than coarse, now on single-mode
# motion too, which is what the fixes bought); it is the absolute percentages
# that are stale.
#
#   FINE     (0.5, 1.0, 0.015)   98.1% mean,  3.2 pp spread
#   BALANCED (1.0, 2.0, 0.05)    89.4% mean,  3.0 pp spread    <-- default
#   COARSE   (2.5, 5.0, 0.1)     72.6% mean, 15.4 pp spread
#
# The pre-PR-#5 pixel-metric values (0.15 / 0.4 / 1.0 for articulation)
# are now re-interpretable as 0.0015 / 0.004 / 0.01 of body-radius under
# the new metric — super tight, keeps everything, hence the retune.
class TolerancePreset(NamedTuple):
    camera_tolerance: float
    person_tolerance: float
    articulation_tolerance: float


PRESET_FINE = TolerancePreset(0.5, 1.0, 0.015)
PRESET_BALANCED = TolerancePreset(1.0, 2.0, 0.05)
PRESET_COARSE = TolerancePreset(2.5, 5.0, 0.1)

# Named-preset lookup for the CLI (and anyone else wanting string IDs).
PRESETS = {
    "fine": PRESET_FINE,
    "balanced": PRESET_BALANCED,
    "coarse": PRESET_COARSE,
}

DEFAULT_CAMERA_TOLERANCE = PRESET_BALANCED.camera_tolerance
DEFAULT_PERSON_TOLERANCE = PRESET_BALANCED.person_tolerance
DEFAULT_ARTICULATION_TOLERANCE = PRESET_BALANCED.articulation_tolerance


def rdp_reduction(points: Sequence[Sequence[float]], tolerance: float) -> List[int]:
    """Ramer-Douglas-Peucker for an N-dimensional polyline SAMPLED IN TIME.

    The error measured is the deviation of each dropped sample from the
    **linear-in-time interpolation** between its retained neighbours -- not
    the perpendicular distance to the chord in state space.

    That distinction is the whole point. These samples are one per video
    frame, and what a host reconstructs between two keyframes is a linear
    ramp **in time**. Measuring perpendicular distance instead treats the
    samples as an unparameterised curve, so motion that retraces its own
    path becomes geometrically indistinguishable from standing still: a
    rigid limb swinging +/-30deg varies only in the bone-angle component,
    traces a 1-D segment back and forth, and every interior sample projects
    exactly onto the chord with zero perpendicular distance. The tolerance
    then has NO EFFECT -- 6 extrema were kept at every preset, and the
    reconstruction was 20.8px out at the wrist on a 4K plate. Pure
    translation and pure uniform deform fail the same way, and so does any
    motion driven by a single time-varying scalar.

    With the time axis respected, ``tolerance`` means what this module has
    always documented: "the remaining keyframes still describe the full
    motion to within ``tolerance``".

    Args:
        points: ordered sequence of state vectors, one per frame -- each a
            sequence of floats, all the same length. ORDER IS TIME, and
            samples are assumed evenly spaced (one per frame).
        tolerance: maximum deviation (in state-vector units) of any dropped
            sample from the linear-in-time interpolation between the
            retained samples either side of it.

    Returns:
        Sorted list of indices into ``points`` for the retained vertices.
        Always includes the first and last indices when ``points`` is
        non-empty.
    """
    n = len(points)
    if n == 0:
        return []
    if n < 3:
        return list(range(n))
    if tolerance <= 0:
        return list(range(n))

    width = len(points[0])
    keep = [0, n - 1]
    # Iterative rather than recursive: a long clip is thousands of frames
    # and the recursive form could reach Python's stack limit on a signal
    # that splits all the way down.
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        a = points[lo]
        b = points[hi]
        span = hi - lo
        worst = 0.0
        pivot = -1
        for i in range(lo + 1, hi):
            t = (i - lo) / span
            p = points[i]
            acc = 0.0
            for j in range(width):
                d = p[j] - (a[j] + t * (b[j] - a[j]))
                acc += d * d
            dist = math.sqrt(acc)
            if dist > worst:
                worst = dist
                pivot = i
        if worst > tolerance and pivot > 0:
            keep.append(pivot)
            stack.append((lo, pivot))
            stack.append((pivot, hi))

    return sorted(keep)


def _state_vector_for_frame(
    frame: LozengeFrame,
    pelvis_px: Optional[Tuple[float, float]] = None,
    body_scale: float = 1.0,
    angle_radians: bool = False,
) -> Sequence[float] | None:
    """Encode one frame as an N-dim state vector.

    Shape: ``[OriginX, OriginY, RotationDegrees, lx_0, ly_0, ..., lx_N, ly_N]``
    where ``(OriginX, OriginY) == bone.pt0`` and local coords are
    computed in a frame-aligned basis built from ``bone.pt1 - bone.pt0``.

    When ``pelvis_px`` is supplied (issue #279 hierarchical path), the
    bone origin is reported relative to the person's pelvis pixel
    position so the first two components of the state vector carry
    articulation-only drift.

    When ``body_scale > 0`` is supplied, every spatial component (bone
    origin AND local knot coords) is divided by it so the state vector
    becomes **dimensionless, body-local**. The intended value is the
    person's max tapered-capsule radius (:func:`_person_max_radius`),
    which shrinks with camera distance exactly as the actor does — a
    tolerance expressed in units of ``body_scale`` means the same
    "amount of articulation" for a near actor and a far actor in the
    same shot, and the same across body parts of one person.

    ``angle_radians`` selects the unit of the rotation component, and
    defaults to DEGREES because that is what the legacy pixel metric was
    tuned against. The hierarchical path passes True, because degrees are
    not commensurate with the dimensionless spatial terms beside them: an
    angle in radians times the body radius IS the arc displacement it
    produces at the body's edge, so in body-radius units the radian value
    already equals that displacement. Degrees overstate the same motion by
    57x, which let the rotation component saturate the metric whenever a
    bone rotated at all and made the body-radius normalisation decorative
    -- with an articulation tolerance of 0.015-0.1, degrees meant "a
    twentieth of a degree", which no real bone satisfies, so every preset
    kept the same frames.

    Returns ``None`` if the frame has no ``bone`` or no points.
    """
    if frame.bone is None or not frame.points:
        return None

    (p0x, p0y), (p1x, p1y) = frame.bone
    if pelvis_px is not None:
        p0x -= pelvis_px[0]
        p0y -= pelvis_px[1]
        p1x -= pelvis_px[0]
        p1y -= pelvis_px[1]
    dx = p1x - p0x
    dy = p1y - p0y
    dist = math.hypot(dx, dy)
    if dist > 0.0:
        ux = (dx / dist, dy / dist)
    else:
        ux = (1.0, 0.0)
    uy = (-ux[1], ux[0])
    # Negated to match the port source — direction of frame rotation
    # relative to bone-local X.
    angle = -math.atan2(dy, dx)
    if not angle_radians:
        angle = math.degrees(angle)

    s = body_scale if body_scale > 0 else 1.0

    out: List[float] = [p0x / s, p0y / s, angle]
    for p in frame.points:
        # Use the ORIGINAL (not pelvis-subtracted) p coords for local
        # projection — pelvis subtraction cancels out of relative
        # coordinates, so this is equivalent to subtracting from both and
        # then taking the difference. Avoids a double-subtract.
        rel_x = p.x - (p0x + (pelvis_px[0] if pelvis_px is not None else 0.0))
        rel_y = p.y - (p0y + (pelvis_px[1] if pelvis_px is not None else 0.0))
        lx = rel_x * ux[0] + rel_y * ux[1]
        ly = rel_x * uy[0] + rel_y * uy[1]
        out.append(lx / s)
        out.append(ly / s)
    return out


def _person_max_radius(doc: LozengeDoc) -> dict:
    """Per-person maximum tapered-capsule radius, in plate pixels.

    For each person, walk every one of their lozenge objects + every
    frame of each + every control point of each; compute the point's
    perpendicular distance to its bone's axis. Return the max over all
    of a person's objects, keyed by ``person_id``.

    This is a clip-wide scalar (one value per person for the whole
    doc), not per-frame — a stable normaliser that reflects how large
    the actor projects onto the plate. On far actors the max shrinks;
    on near actors it grows; same for both at the SAME frame.

    Returns an empty dict when the doc has no objects or no points.
    """
    out: dict = {}
    for obj in doc.objects.values():
        if not obj.frames:
            continue
        pid = obj.person_id
        for frame in obj.frames.values():
            if frame.bone is None or not frame.points:
                continue
            (p0x, p0y), (p1x, p1y) = frame.bone
            dx = p1x - p0x
            dy = p1y - p0y
            dist = math.hypot(dx, dy)
            if dist <= 0.0:
                continue
            # Perpendicular unit vector to the bone axis. The lozenge
            # radius at any knot is |projection of (knot - p0) onto uy|.
            uyx = -dy / dist
            uyy = dx / dist
            for p in frame.points:
                rx = p.x - p0x
                ry = p.y - p0y
                perp = abs(rx * uyx + ry * uyy)
                if perp > out.get(pid, 0.0):
                    out[pid] = perp
    return out


def _camera_state_vector(cam: CameraFrame) -> Sequence[float]:
    """Encode a plate-camera reference as a 21-dim state vector.

    Shape: ``[tx, ty, tz] ++ R[9] ++ H2d[9]``. Identity matrices stand in
    when the source didn't supply ``R`` or ``H2d`` (locked-off shots,
    sidecars that only provided intrinsics, etc).
    """
    out: List[float] = [cam.t[0], cam.t[1], cam.t[2]]
    out.extend(cam.R if cam.R is not None else (1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0))
    out.extend(cam.H2d if cam.H2d is not None else (1.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 0.0, 1.0))
    return out


def _person_state_vector(pf: PersonFrame) -> Sequence[float]:
    """``[cam_t.x, cam_t.y, cam_t.z, pelvis_px.x, pelvis_px.y]`` — 5 dims."""
    return [pf.cam_t[0], pf.cam_t[1], pf.cam_t[2],
            pf.pelvis_px[0], pf.pelvis_px[1]]


def undersample_object(
    obj: LozengeObject, tolerance: float = DEFAULT_TOLERANCE
) -> Tuple[LozengeObject, int, int]:
    """Return a copy of ``obj`` with redundant keyframes dropped.

    Objects without per-frame ``bone`` data (schema v1, or v2 captures
    that didn't record it) are returned unchanged — RDP needs the bone
    anchor to produce a meaningful state vector.

    Visibility keys are kept intact; a hidden-frame key is a real
    annotation of what the artist should see, not a candidate for
    reduction.

    Returns:
        ``(new_object, frames_before, frames_after)``.
    """
    if not obj.frames or tolerance <= 0:
        return obj, len(obj.frames), len(obj.frames)

    sorted_keys = sorted(obj.frames.keys())
    state_vectors: List[Sequence[float]] = []
    for k in sorted_keys:
        sv = _state_vector_for_frame(obj.frames[k])
        if sv is None:
            # Can't compare without a bone; bail out and keep every
            # frame so we don't silently corrupt timing.
            return obj, len(obj.frames), len(obj.frames)
        state_vectors.append(sv)

    # Enforce uniform width — if the point-count-per-frame varies, pad or
    # bail. Rotobot-Next emits constant point_count per object, so a
    # mismatch means malformed input; keep everything.
    widths = {len(sv) for sv in state_vectors}
    if len(widths) != 1:
        return obj, len(obj.frames), len(obj.frames)

    keep_indices = set(rdp_reduction(state_vectors, tolerance))
    kept_keys = [sorted_keys[i] for i in sorted(keep_indices)]
    new_frames = {k: obj.frames[k] for k in kept_keys}

    new_obj = replace(obj, frames=new_frames)
    return new_obj, len(obj.frames), len(new_frames)


def _hierarchical_undersample(
    doc: LozengeDoc,
    *,
    camera_tolerance: float,
    person_tolerance: float,
    articulation_tolerance: float,
) -> LozengeDoc:
    """Three-pass composed-tolerance keyframe reduction (issue #279).

    Pass 1: RDP over :attr:`LozengeDoc.camera` under ``camera_tolerance``.
            Camera + person state vectors stay in screen pixels (metres
            for the ``t`` / ``cam_t`` components); these passes encode
            "where on the plate to draw the roto", so pixel units match
            the thing being measured.
    Pass 2: RDP over :attr:`LozengeDoc.persons` per pid under ``person_tolerance``.
    Pass 3: RDP over each object's articulation-only state vector
            (bone origin minus pelvis) under ``articulation_tolerance``.
            Articulation vectors are normalised by the person's max
            tapered-capsule radius so the tolerance is a dimensionless
            fraction of body-radius — depth-invariant per person,
            cross-body-part consistent within one person.

    Final kept frames per object = union of the three passes' retained
    frames, intersected with the object's own frame set.
    """
    # Clip-wide per-person body scale (max tapered-capsule radius in
    # plate pixels). Falls back to 1.0 when no data is available for a
    # given person — in that case the articulation vectors stay in
    # pixel units, exactly as PR #2 shipped them.
    person_scale = _person_max_radius(doc)

    # Pass 1 — camera.
    kept_cam_frames: set = set()
    if doc.camera:
        cam_frames_sorted = sorted(doc.camera.keys())
        if camera_tolerance > 0 and len(cam_frames_sorted) >= 2:
            cam_vectors = [_camera_state_vector(doc.camera[f]) for f in cam_frames_sorted]
            kept_idx = rdp_reduction(cam_vectors, camera_tolerance)
            kept_cam_frames = {cam_frames_sorted[i] for i in kept_idx}
        else:
            kept_cam_frames = set(cam_frames_sorted)

    # Pass 2 — person root per pid. Pivot the frame-major persons dict
    # into pid-major so each pid gets its own time-series.
    kept_person_frames_by_pid: dict = {}
    if doc.persons:
        persons_by_pid: dict = {}
        for f, per_frame in doc.persons.items():
            for pid, pf in per_frame.items():
                persons_by_pid.setdefault(pid, {})[f] = pf
        for pid, frames in persons_by_pid.items():
            sorted_frames = sorted(frames.keys())
            if person_tolerance > 0 and len(sorted_frames) >= 2:
                vectors = [_person_state_vector(frames[f]) for f in sorted_frames]
                kept_idx = rdp_reduction(vectors, person_tolerance)
                kept_person_frames_by_pid[pid] = {sorted_frames[i] for i in kept_idx}
            else:
                kept_person_frames_by_pid[pid] = set(sorted_frames)

    # Pass 3 — per-object articulation with per-frame pelvis subtracted.
    new_objects: dict = {}
    for key, obj in doc.objects.items():
        if not obj.frames:
            new_objects[key] = obj
            continue
        sorted_frames = sorted(obj.frames.keys())
        pid = obj.person_id

        # Collect per-frame pelvis_px (may be empty if persons data absent).
        pelvis_by_frame: dict = {}
        if doc.persons:
            for f in sorted_frames:
                per = doc.persons.get(f)
                if per is None:
                    continue
                pf = per.get(pid)
                if pf is not None:
                    pelvis_by_frame[f] = pf.pelvis_px

        # Build the articulation state vectors. If any frame lacks a bone,
        # bail on this object — can't compare without the anchor.
        # body_scale normalises the vector to body-local fractions so a
        # fixed `articulation_tolerance` means the same amount of
        # articulation regardless of how far the actor is from camera.
        scale = person_scale.get(pid, 1.0)
        articulation_vectors: list = []
        bone_ok = True
        for f in sorted_frames:
            sv = _state_vector_for_frame(
                obj.frames[f],
                pelvis_px=pelvis_by_frame.get(f),
                body_scale=scale,
                angle_radians=True,
            )
            if sv is None:
                bone_ok = False
                break
            articulation_vectors.append(sv)

        if not bone_ok or len({len(v) for v in articulation_vectors}) != 1:
            # Bone-less or varying-K object — keep every frame on the
            # articulation pass; the outer passes still prune via the union.
            kept_art_frames: set = set(sorted_frames)
        elif articulation_tolerance <= 0 or len(articulation_vectors) < 3:
            kept_art_frames = set(sorted_frames)
        else:
            kept_idx = rdp_reduction(articulation_vectors, articulation_tolerance)
            kept_art_frames = {sorted_frames[i] for i in kept_idx}

        # Compose: a frame is kept whenever ANY of the three passes
        # retained it (and it exists on this object).
        kept_person_frames = kept_person_frames_by_pid.get(pid, set())
        obj_frames_set = set(sorted_frames)
        final_kept = (kept_cam_frames | kept_person_frames | kept_art_frames) & obj_frames_set
        if not final_kept:
            # Degenerate: no outer-pass frames landed on this object AND
            # the object has only 1-2 articulation frames. Keep everything.
            final_kept = obj_frames_set

        new_frames = {f: obj.frames[f] for f in sorted(final_kept)}
        new_objects[key] = replace(obj, frames=new_frames)

    return replace(doc, objects=new_objects)


def undersample_doc(
    doc: LozengeDoc,
    tolerance: Optional[float] = None,
    *,
    camera_tolerance: Optional[float] = None,
    person_tolerance: Optional[float] = None,
    articulation_tolerance: Optional[float] = None,
) -> LozengeDoc:
    """Return a copy of ``doc`` with each object's keyframes RDP-reduced.

    Two call shapes:

    * **Legacy single-pass** (PR #1): pass ``tolerance=X`` or no kwargs.
      One bone-local RDP per object; identical behaviour to v0.1.
    * **Hierarchical composed-tolerance** (issue #279): pass any of
      ``camera_tolerance``, ``person_tolerance``, ``articulation_tolerance``.
      Three sequential RDP passes; missing kwargs fall back to
      :data:`PRESET_BALANCED`. See :func:`_hierarchical_undersample`.

    Shape geometry is untouched — only which frames carry a keyframe
    changes. Combine with :func:`rotobot_nuke.importer.build_roto` to
    turn the reduced doc into a Roto node.
    """
    hierarchical = any(
        x is not None
        for x in (camera_tolerance, person_tolerance, articulation_tolerance)
    )
    if hierarchical:
        if tolerance is not None:
            raise TypeError(
                "undersample_doc: pass EITHER tolerance= (legacy single-pass) "
                "OR the hierarchical kwargs (camera_tolerance / person_tolerance "
                "/ articulation_tolerance), not both."
            )
        return _hierarchical_undersample(
            doc,
            camera_tolerance=camera_tolerance
            if camera_tolerance is not None
            else DEFAULT_CAMERA_TOLERANCE,
            person_tolerance=person_tolerance
            if person_tolerance is not None
            else DEFAULT_PERSON_TOLERANCE,
            articulation_tolerance=articulation_tolerance
            if articulation_tolerance is not None
            else DEFAULT_ARTICULATION_TOLERANCE,
        )

    # Legacy single-pass path.
    tol = DEFAULT_TOLERANCE if tolerance is None else tolerance
    if tol <= 0:
        return doc

    new_objects: dict = {}
    for key, obj in doc.objects.items():
        new_obj, _before, _after = undersample_object(obj, tol)
        new_objects[key] = new_obj

    return replace(doc, objects=new_objects)


def undersample_json(
    input_path,
    output_path,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    camera_tolerance: Optional[float] = None,
    person_tolerance: Optional[float] = None,
    articulation_tolerance: Optional[float] = None,
) -> dict:
    """File-to-file convenience: load ``input_path``, undersample,
    write to ``output_path``. Returns a per-object before/after report.

    Two call shapes, matching :func:`undersample_doc`:

    * **Legacy single-pass**: pass ``tolerance=X`` or no hierarchical
      kwargs. Runs the pixel-metric bone-local RDP per object; the
      operator-facing conversion that shipped in PR #1. Preserves the
      exact JSON top-level structure byte-for-byte beyond the filtered
      ``frames`` dicts.

    * **Hierarchical composed-tolerance** (issue #279): pass any of
      ``camera_tolerance``, ``person_tolerance``, ``articulation_tolerance``.
      Routes through :func:`undersample_doc`, which uses the v3
      ``camera`` + ``persons`` blocks and the body-local articulation
      metric. The output JSON is reserialised from the parsed
      :class:`LozengeDoc`, so key order within each object matches the
      current writer convention (not necessarily the input).

    Writes `frames` whose keys weren't retained removed; nothing else
    is renamed or reshaped.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    hierarchical = any(
        x is not None
        for x in (camera_tolerance, person_tolerance, articulation_tolerance)
    )

    if hierarchical:
        # Route through the parsed-doc hierarchical path. The output
        # JSON then needs reserialising from the raw dict's structure
        # with filtered frame sets.
        from .reader import load_json as _load_json  # local to keep CLI light

        doc = _load_json(in_path)
        reduced = undersample_doc(
            doc,
            camera_tolerance=camera_tolerance,
            person_tolerance=person_tolerance,
            articulation_tolerance=articulation_tolerance,
        )

        with in_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        objects = raw.get("objects") or {}
        report: dict = {}
        for obj_id, obj_raw in list(objects.items()):
            frames_raw = obj_raw.get("frames") or {}
            kept_obj = reduced.objects.get(obj_id)
            if kept_obj is None:
                report[obj_id] = (len(frames_raw), 0)
                obj_raw["frames"] = {}
                continue
            kept_keys = {str(k) for k in kept_obj.frames.keys()} | {
                str(k).lstrip("0") or "0" for k in kept_obj.frames.keys()
            }
            # Preserve the raw key representations the input used (string
            # frame keys could be "0", "00", or "0001" etc.).
            new_frames = {}
            for raw_k, raw_v in frames_raw.items():
                try:
                    kept = int(raw_k) in kept_obj.frames
                except (TypeError, ValueError):
                    kept = raw_k in kept_keys
                if kept:
                    new_frames[raw_k] = raw_v
            report[obj_id] = (len(frames_raw), len(new_frames))
            obj_raw["frames"] = new_frames

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2)
        return report

    # Legacy single-pass path (unchanged).
    with in_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    objects = raw.get("objects") or {}
    report: dict = {}

    # Need bones to run RDP — but we also need the raw frame dict so
    # we can keep the original indentation / key order. Walk the raw
    # dict, build a LozengeDoc-equivalent view per object, then filter
    # the raw frames dict in-place.
    for obj_id, obj_raw in list(objects.items()):
        frames_raw = obj_raw.get("frames") or {}
        if not frames_raw:
            report[obj_id] = (0, 0)
            continue

        try:
            sorted_keys = sorted(frames_raw.keys(), key=lambda s: int(s))
        except (TypeError, ValueError):
            report[obj_id] = (len(frames_raw), len(frames_raw))
            continue

        state_vectors: List[Sequence[float]] = []
        bail = False
        for k in sorted_keys:
            fv = frames_raw[k]
            if not isinstance(fv, dict):
                bail = True
                break
            bone = fv.get("bone")
            pts = fv.get("points")
            if not isinstance(bone, dict) or not isinstance(pts, list):
                bail = True
                break
            pt0 = bone.get("pt0") or {}
            pt1 = bone.get("pt1") or {}
            try:
                p0x = float(pt0["x"])
                p0y = float(pt0["y"])
                p1x = float(pt1["x"])
                p1y = float(pt1["y"])
            except (KeyError, TypeError, ValueError):
                bail = True
                break
            dx = p1x - p0x
            dy = p1y - p0y
            dist = math.hypot(dx, dy)
            if dist > 0.0:
                ux = (dx / dist, dy / dist)
            else:
                ux = (1.0, 0.0)
            uy = (-ux[1], ux[0])
            angle = -math.degrees(math.atan2(dy, dx))
            sv: List[float] = [p0x, p0y, angle]
            try:
                for p in pts:
                    rx = float(p["x"]) - p0x
                    ry = float(p["y"]) - p0y
                    sv.append(rx * ux[0] + ry * ux[1])
                    sv.append(rx * uy[0] + ry * uy[1])
            except (KeyError, TypeError, ValueError):
                bail = True
                break
            state_vectors.append(sv)

        if bail or len({len(sv) for sv in state_vectors}) != 1:
            report[obj_id] = (len(frames_raw), len(frames_raw))
            continue

        keep_indices = set(rdp_reduction(state_vectors, tolerance))
        keep_keys = {sorted_keys[i] for i in keep_indices}
        obj_raw["frames"] = {k: v for k, v in frames_raw.items() if k in keep_keys}
        report[obj_id] = (len(frames_raw), len(obj_raw["frames"]))

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)

    return report


def _cli(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rotobot-undersample",
        description=(
            "Reduce the per-object temporal keyframe count in a Rotobot-Next "
            "lozenge_bezier_anim JSON. Legacy single-pass RDP by default; "
            "hierarchical composed-tolerance (PR #5 body-local metric) via "
            "--preset or individual --*-tolerance kwargs."
        ),
    )
    parser.add_argument("input", help="input JSON path")
    parser.add_argument("output", help="output JSON path")

    legacy = parser.add_argument_group("legacy single-pass")
    legacy.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=None,
        help=(
            "Single-pass RDP tolerance in pixel-equivalents. Backwards-"
            f"compatible with PR #1 (default {DEFAULT_TOLERANCE} when neither "
            "a preset nor hierarchical kwargs are passed)."
        ),
    )

    hier = parser.add_argument_group(
        "hierarchical composed-tolerance (v3, body-local)"
    )
    hier.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help=(
            "Named triple (fine/balanced/coarse) from the 4-clip "
            "cross-clip sweep in benchmarks/real_results_cross_clip.md. "
            "Individual --*-tolerance kwargs override preset components."
        ),
    )
    hier.add_argument(
        "--camera-tolerance",
        type=float,
        default=None,
        help="Pass-1 (camera H2d) tolerance in screen pixels.",
    )
    hier.add_argument(
        "--person-tolerance",
        type=float,
        default=None,
        help="Pass-2 (person root) tolerance in screen pixels.",
    )
    hier.add_argument(
        "--articulation-tolerance",
        type=float,
        default=None,
        help=(
            "Pass-3 (per-object articulation) tolerance as a fraction of "
            "the person's max tapered-capsule radius (body-local). PR #5."
        ),
    )

    args = parser.parse_args(argv)

    # Resolve the preset, letting individual kwargs override.
    camera_tol = args.camera_tolerance
    person_tol = args.person_tolerance
    articulation_tol = args.articulation_tolerance
    if args.preset is not None:
        p = PRESETS[args.preset]
        if camera_tol is None:
            camera_tol = p.camera_tolerance
        if person_tol is None:
            person_tol = p.person_tolerance
        if articulation_tol is None:
            articulation_tol = p.articulation_tolerance

    hierarchical = any(
        x is not None for x in (camera_tol, person_tol, articulation_tol)
    )

    if hierarchical and args.tolerance is not None:
        parser.error(
            "pass EITHER --tolerance (legacy single-pass) OR a preset / "
            "hierarchical kwargs, not both"
        )

    if hierarchical:
        report = undersample_json(
            args.input,
            args.output,
            camera_tolerance=camera_tol,
            person_tolerance=person_tol,
            articulation_tolerance=articulation_tol,
        )
        label = (
            f"hierarchical (cam={camera_tol}, person={person_tol}, "
            f"articulation={articulation_tol})"
        )
    else:
        tol = args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE
        report = undersample_json(args.input, args.output, tolerance=tol)
        label = f"single-pass tolerance={tol}"

    for obj_id, (before, after) in sorted(report.items()):
        pct = (100.0 * after / before) if before else 0.0
        print(
            f"{obj_id}: {before} -> {after} frames ({pct:.1f}% kept)"
        )
    kept = sum(a for _, a in report.values())
    orig = sum(b for b, _ in report.values())
    if orig:
        print(
            f"TOTAL: {orig} -> {kept} frames "
            f"({100.0 * kept / orig:.1f}% kept) at {label}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
