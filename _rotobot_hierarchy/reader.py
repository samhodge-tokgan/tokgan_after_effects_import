"""Pure Python parser for the Rotobot-Next ``lozenge_bezier_anim`` schema.

No ``nuke`` import; safe to use (and test) outside of Nuke.

Supported schema versions:
    v3 — adds optional top-level ``camera`` + ``persons`` reference-frame
         blocks (issue #279) so the hierarchical undersampler can subtract
         plate motion + whole-body translation before measuring articulation.
    v2 — carries ``resolution``/``width``/``height``, per-object ``visibility``,
         per-frame ``bone``, optional ``person_depth``.
    v1 — legacy; lacks any in-band resolution. The ``resolution`` kwarg is
         required when loading these.

Resolution policy (per ``load_json``):
    1. explicit ``resolution=(w, h)`` kwarg wins
    2. else JSON ``resolution`` / ``height`` / ``width`` wins
    3. else :class:`MissingResolutionError` is raised

The importer never falls back to ``nuke.root()["format"]``; the artist's
project settings do not have to match the plate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping, Optional, Tuple, Union

SCHEMA_NAME = "lozenge_bezier_anim"

PathLike = Union[str, Path, IO[str]]


class MissingResolutionError(ValueError):
    """Raised when the plate resolution cannot be determined.

    Message always names the ``resolution=(w, h)`` kwarg so the caller
    knows how to supply it.
    """


@dataclass(frozen=True)
class LozengePoint:
    """One cubic-Bezier knot: anchor + two absolute tangent-handle positions.

    Coordinates are **absolute pixels, Y-down** (image/OpenCV convention).
    The Nuke-side importer handles the flip and the handle-to-delta
    conversion.
    """

    x: float
    y: float
    left_x: float
    left_y: float
    right_x: float
    right_y: float


@dataclass
class LozengeFrame:
    points: list
    bone: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


@dataclass
class LozengeObject:
    """One anatomical segment (e.g. ``p0:leg:R:thigh``) across all frames."""

    key: str
    person_id: int
    body: str
    side: str
    segment: str
    skeleton_edge: Optional[Tuple[int, int]]
    closed: bool
    point_count: int
    visibility: dict
    frames: dict


@dataclass(frozen=True)
class CameraFrame:
    """v3 (#279) per-frame plate-camera reference. All fields optional on the
    wire — unspecified values default to identity transforms.

    Coordinates are MHR camera space (metres, +Z forward) for ``t`` + ``R``
    and plate pixels for ``H2d``. ``source`` names the provenance of this
    entry (``sidecar`` / ``ecc`` / ``identity``).
    """

    focal: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    t: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R: Optional[Tuple[float, ...]] = None  # 9 floats, 3x3 row-major
    H2d: Optional[Tuple[float, ...]] = None  # 9 floats, 3x3 pixel homography
    source: str = ""


@dataclass(frozen=True)
class PersonFrame:
    """v3 (#279) per-frame per-person root pose.

    ``pelvis_px`` is the mhr70 kp 9/10 (L+R hip) midpoint in plate pixels.
    ``pelvis_3d`` is the camera-space 3D midpoint when 3D keypoints exist.
    ``joint_xforms`` and ``pose`` are passed through verbatim from the Hastur
    sidecar when it recorded them; nothing in ``rotobot-nuke`` interprets
    their contents, but they travel with the doc for third-party consumers.
    """

    cam_t: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pelvis_px: Tuple[float, float] = (0.0, 0.0)
    pelvis_3d: Optional[Tuple[float, float, float]] = None
    joint_xforms: Tuple[float, ...] = ()
    pose: Tuple[float, ...] = ()


@dataclass
class LozengeDoc:
    schema: str
    schema_version: int
    fps: float
    resolution: Tuple[int, int]
    objects: dict
    person_depth: Optional[dict] = None
    # v3 (#279): frame -> CameraFrame
    camera: Optional[dict] = None
    # v3 (#279): frame -> pid -> PersonFrame
    persons: Optional[dict] = None


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None or isinstance(x, bool):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _parse_object_key(key: str) -> Tuple[int, str, str, str]:
    """``p0:leg:R:thigh`` -> (0, "leg", "R", "thigh"). Tolerates malformed keys."""
    parts = key.split(":")
    while len(parts) < 4:
        parts.append("unknown")
    pid_raw = parts[0]
    person_id = 0
    if pid_raw.startswith("p"):
        pid_num = _as_int(pid_raw[1:])
        if pid_num is not None:
            person_id = pid_num
    return person_id, parts[1], parts[2], parts[3]


def _parse_point(raw: Mapping[str, Any]) -> LozengePoint:
    x = float(raw["x"])
    y = float(raw["y"])
    return LozengePoint(
        x=x,
        y=y,
        left_x=float(raw.get("left_x", x)),
        left_y=float(raw.get("left_y", y)),
        right_x=float(raw.get("right_x", x)),
        right_y=float(raw.get("right_y", y)),
    )


def _parse_frame(raw: Any) -> LozengeFrame:
    """Accept both v2 shape (``{"points": [...], "bone": {...}}``) and v1
    shape (``[{"x":..., "y":..., ...}, ...]`` — a bare point list)."""
    if isinstance(raw, list):
        return LozengeFrame(points=[_parse_point(p) for p in raw])

    if not isinstance(raw, Mapping):
        raise ValueError(f"frame payload is neither list nor object: {type(raw).__name__}")

    pts_raw = raw.get("points")
    if pts_raw is None:
        pts = []
    else:
        pts = [_parse_point(p) for p in pts_raw]

    bone = None
    bone_raw = raw.get("bone")
    if isinstance(bone_raw, Mapping):
        pt0 = bone_raw.get("pt0")
        pt1 = bone_raw.get("pt1")
        if isinstance(pt0, Mapping) and isinstance(pt1, Mapping):
            bone = (
                (float(pt0["x"]), float(pt0["y"])),
                (float(pt1["x"]), float(pt1["y"])),
            )

    return LozengeFrame(points=pts, bone=bone)


def _parse_object(key: str, raw: Mapping[str, Any]) -> LozengeObject:
    person_id = _as_int(raw.get("person_id"))
    body = str(raw.get("body", ""))
    side = str(raw.get("side", ""))
    segment = str(raw.get("segment", ""))

    # Fall back to the key for missing anatomical fields (v1 compat).
    if not (body and side and segment):
        pid_from_key, body_k, side_k, segment_k = _parse_object_key(key)
        if person_id is None:
            person_id = pid_from_key
        body = body or body_k
        side = side or side_k
        segment = segment or segment_k

    if person_id is None:
        person_id, _, _, _ = _parse_object_key(key)

    edge_raw = raw.get("skeleton_edge")
    skeleton_edge: Optional[Tuple[int, int]] = None
    if isinstance(edge_raw, (list, tuple)) and len(edge_raw) == 2:
        j0 = _as_int(edge_raw[0])
        j1 = _as_int(edge_raw[1])
        if j0 is not None and j1 is not None:
            skeleton_edge = (j0, j1)

    closed = bool(raw.get("closed", True))
    point_count = _as_int(raw.get("point_count")) or 0

    vis_raw = raw.get("visibility", {}) or {}
    visibility: dict = {}
    if isinstance(vis_raw, Mapping):
        for fk, fv in vis_raw.items():
            fi = _as_int(fk)
            if fi is None:
                continue
            visibility[fi] = bool(fv)

    frames_raw = raw.get("frames", {}) or {}
    frames: dict = {}
    if isinstance(frames_raw, Mapping):
        for fk, fv in frames_raw.items():
            fi = _as_int(fk)
            if fi is None:
                continue
            frames[fi] = _parse_frame(fv)

    return LozengeObject(
        key=key,
        person_id=person_id,
        body=body,
        side=side,
        segment=segment,
        skeleton_edge=skeleton_edge,
        closed=closed,
        point_count=point_count,
        visibility=visibility,
        frames=frames,
    )


def _resolve_resolution(
    data: Mapping[str, Any], override: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """Resolution-resolution policy. See module docstring."""
    if override is not None:
        try:
            w, h = int(override[0]), int(override[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise MissingResolutionError(
                "resolution kwarg must be a (width, height) pair of ints; "
                f"got {override!r}"
            ) from exc
        if w <= 0 or h <= 0:
            raise MissingResolutionError(
                f"resolution kwarg must be positive; got (w={w}, h={h})"
            )
        return (w, h)

    res = data.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) == 2:
        w = _as_int(res[0])
        h = _as_int(res[1])
        if w and h:
            return (w, h)

    w = _as_int(data.get("width"))
    h = _as_int(data.get("height"))
    if w and h:
        return (w, h)
    # Allow height-only files to be loaded if the caller doesn't need width
    # for anything; but we always want a complete tuple, so if only one is
    # present we still require the kwarg.
    raise MissingResolutionError(
        "This JSON does not record its plate resolution (no 'resolution', "
        "'width'/'height' fields). Pass resolution=(width, height) explicitly, "
        "e.g. load_json(path, resolution=(1920, 1080))."
    )


def _parse_mat3(raw: Any) -> Optional[Tuple[float, ...]]:
    """Parse a 9-float row-major 3x3 matrix. Returns ``None`` on bad shape."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 9:
        return None
    try:
        return tuple(float(raw[i]) for i in range(9))
    except (TypeError, ValueError):
        return None


def _parse_camera_frame(raw: Any) -> Optional[CameraFrame]:
    if not isinstance(raw, Mapping):
        return None
    t_raw = raw.get("t")
    t: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    if isinstance(t_raw, (list, tuple)) and len(t_raw) >= 3:
        try:
            t = (float(t_raw[0]), float(t_raw[1]), float(t_raw[2]))
        except (TypeError, ValueError):
            t = (0.0, 0.0, 0.0)
    try:
        focal = float(raw.get("focal", 0.0))
        cx = float(raw.get("cx", 0.0))
        cy = float(raw.get("cy", 0.0))
    except (TypeError, ValueError):
        focal = cx = cy = 0.0
    return CameraFrame(
        focal=focal,
        cx=cx,
        cy=cy,
        t=t,
        R=_parse_mat3(raw.get("R")),
        H2d=_parse_mat3(raw.get("H2d")),
        source=str(raw.get("source", "")),
    )


def _parse_person_frame(raw: Any) -> Optional[PersonFrame]:
    if not isinstance(raw, Mapping):
        return None
    cam_t_raw = raw.get("cam_t")
    cam_t: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    if isinstance(cam_t_raw, (list, tuple)) and len(cam_t_raw) >= 3:
        try:
            cam_t = (float(cam_t_raw[0]), float(cam_t_raw[1]), float(cam_t_raw[2]))
        except (TypeError, ValueError):
            cam_t = (0.0, 0.0, 0.0)
    pelvis_px_raw = raw.get("pelvis_px")
    pelvis_px: Tuple[float, float] = (0.0, 0.0)
    if isinstance(pelvis_px_raw, (list, tuple)) and len(pelvis_px_raw) >= 2:
        try:
            pelvis_px = (float(pelvis_px_raw[0]), float(pelvis_px_raw[1]))
        except (TypeError, ValueError):
            pelvis_px = (0.0, 0.0)
    pelvis_3d: Optional[Tuple[float, float, float]] = None
    pelvis_3d_raw = raw.get("pelvis_3d")
    if isinstance(pelvis_3d_raw, (list, tuple)) and len(pelvis_3d_raw) >= 3:
        try:
            pelvis_3d = (
                float(pelvis_3d_raw[0]),
                float(pelvis_3d_raw[1]),
                float(pelvis_3d_raw[2]),
            )
        except (TypeError, ValueError):
            pelvis_3d = None

    def _flat_floats(x) -> Tuple[float, ...]:
        if not isinstance(x, (list, tuple)):
            return ()
        try:
            return tuple(float(v) for v in x)
        except (TypeError, ValueError):
            return ()

    return PersonFrame(
        cam_t=cam_t,
        pelvis_px=pelvis_px,
        pelvis_3d=pelvis_3d,
        joint_xforms=_flat_floats(raw.get("joint_xforms")),
        pose=_flat_floats(raw.get("pose")),
    )


def _parse_camera(raw: Any) -> Optional[dict]:
    if not isinstance(raw, Mapping):
        return None
    out: dict = {}
    for fk, fv in raw.items():
        fi = _as_int(fk)
        if fi is None:
            continue
        parsed = _parse_camera_frame(fv)
        if parsed is None:
            continue
        out[fi] = parsed
    return out or None


def _parse_persons(raw: Any) -> Optional[dict]:
    if not isinstance(raw, Mapping):
        return None
    out: dict = {}
    for fk, fv in raw.items():
        fi = _as_int(fk)
        if fi is None or not isinstance(fv, Mapping):
            continue
        inner: dict = {}
        for pk, pv in fv.items():
            pi = _as_int(pk)
            if pi is None:
                continue
            parsed = _parse_person_frame(pv)
            if parsed is None:
                continue
            inner[pi] = parsed
        if inner:
            out[fi] = inner
    return out or None


def _read_json(path_or_file: PathLike) -> Any:
    if hasattr(path_or_file, "read"):
        return json.load(path_or_file)  # type: ignore[arg-type]
    path = Path(path_or_file)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc


def load_json(
    path_or_file: PathLike,
    *,
    resolution: Optional[Tuple[int, int]] = None,
) -> LozengeDoc:
    """Parse a Rotobot-Next ``lozenge_bezier_anim`` JSON into a :class:`LozengeDoc`.

    Args:
        path_or_file: filesystem path, ``pathlib.Path``, or a text file handle.
        resolution: optional ``(width, height)`` override; wins over anything
            the JSON records. Required for v1 JSONs (no in-band resolution).

    Raises:
        ValueError: the file isn't valid JSON, or has a wrong top-level schema.
        MissingResolutionError: resolution couldn't be determined and no
            kwarg was supplied.
    """
    data = _read_json(path_or_file)
    if not isinstance(data, Mapping):
        raise ValueError(
            f"Top-level JSON must be an object; got {type(data).__name__}"
        )

    schema = str(data.get("schema", ""))
    if schema and schema != SCHEMA_NAME:
        raise ValueError(
            f"Unsupported JSON schema {schema!r}; expected {SCHEMA_NAME!r}"
        )

    schema_version = _as_int(data.get("schema_version")) or 1
    fps_raw = data.get("fps", 24)
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError):
        fps = 24.0

    res = _resolve_resolution(data, resolution)

    objects_raw = data.get("objects", {}) or {}
    objects: dict = {}
    if isinstance(objects_raw, Mapping):
        for key, raw in objects_raw.items():
            if not isinstance(raw, Mapping):
                continue
            objects[key] = _parse_object(key, raw)

    person_depth = None
    pd_raw = data.get("person_depth")
    if isinstance(pd_raw, Mapping):
        person_depth = {}
        for fk, fv in pd_raw.items():
            fi = _as_int(fk)
            if fi is None or not isinstance(fv, Mapping):
                continue
            inner: dict = {}
            for pk, pv in fv.items():
                pi = _as_int(pk)
                if pi is None:
                    continue
                try:
                    inner[pi] = float(pv)
                except (TypeError, ValueError):
                    continue
            person_depth[fi] = inner

    camera = _parse_camera(data.get("camera"))
    persons = _parse_persons(data.get("persons"))

    return LozengeDoc(
        schema=schema or SCHEMA_NAME,
        schema_version=schema_version,
        fps=fps,
        resolution=res,
        objects=objects,
        person_depth=person_depth,
        camera=camera,
        persons=persons,
    )
