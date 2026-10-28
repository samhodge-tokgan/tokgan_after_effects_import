"""Converter tests: flat and hierarchy payloads, and the hierarchy loader run
end to end under a mock of the After Effects scripting DOM (tests/ae_mock.js).

``v3_real_cut.json`` is cut from a real Rotobot Next 0.10.0 run: two body
parts over 24 4K frames with the real (projective) camera and pelvis track.
``v2_cut.json`` is the same data without the camera and persons blocks.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURES = HERE / "fixtures"
SCRIPT = REPO / "tokgan_json_to_ae.py"

sys.path.insert(0, str(REPO))
import tokgan_json_to_ae as conv  # noqa: E402


def run(tmp_path, fixture, *args):
    src = tmp_path / "clip.json"
    shutil.copy(FIXTURES / fixture, src)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(src), *args],
        capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    jsx = tmp_path / "clip.jsx"
    data = json.loads((tmp_path / "clip_data.json").read_text())
    return src, jsx, data, out.stdout


class TestModes:
    def test_v3_builds_the_hierarchy(self, tmp_path):
        _, jsx, data, stdout = run(tmp_path, "v3_real_cut.json")
        assert data["mode"] == "hierarchy"
        assert "Tokgan Stabilised" in jsx.read_text()
        assert "Hierarchy: 1 person Null(s), 2 part layers" in stdout

    def test_flat_flag_keeps_the_single_layer(self, tmp_path):
        _, jsx, data, _ = run(tmp_path, "v3_real_cut.json", "--flat")
        assert data["mode"] == "flat"
        assert "Tokgan Stabilised" not in jsx.read_text()

    def test_v2_is_flat(self, tmp_path):
        _, _, data, _ = run(tmp_path, "v2_cut.json")
        assert data["mode"] == "flat"

    def test_flat_loader_is_unchanged_by_the_hierarchy_work(self, tmp_path):
        _, jsx, _, _ = run(tmp_path, "v2_cut.json")
        # The flat loader is the template with the sidecar name and the timing code
        # shared by every loader (Rotobot Next v0.20: time zero is the clip's first frame).
        assert jsx.read_text() == conv.LOADER_TEMPLATE.format(data_basename="clip_data.json",
                                                             common=conv.COMMON_JS)

    def test_switching_mode_rebuilds_the_sidecar(self, tmp_path):
        run(tmp_path, "v3_real_cut.json")
        src = tmp_path / "clip.json"
        out = subprocess.run([sys.executable, str(SCRIPT), str(src), "--flat"],
                             capture_output=True, text=True)
        assert "Reusing" not in out.stdout
        assert json.loads((tmp_path / "clip_data.json").read_text())["mode"] == "flat"


class TestHierarchyPayload:
    def test_structure(self, tmp_path):
        _, _, data, _ = run(tmp_path, "v3_real_cut.json")
        assert len(data["camera"]["times"]) == 24
        assert all(len(c) == 4 for c in data["camera"]["corners"])
        (person,) = data["persons"]
        assert person["name"] == "p0_pelvis" and len(person["pos"]) == 24
        for s in data["shapes"]:
            n = len(s["times"])
            assert len(s["verts"]) == len(s["xf"]["pos"]) == len(s["xf"]["rot"]) == n

    def test_precomp_holds_every_stabilised_vertex(self, tmp_path):
        _, _, data, _ = run(tmp_path, "v3_real_cut.json")
        pre = data["precomp"]
        assert pre["width"] >= 3840 and pre["height"] >= 2160
        assert pre["width"] <= conv.AE_MAX_COMP_SIZE

    def test_camera_is_projective(self, tmp_path):
        """The corner pin is not a parallelogram: perspective was kept."""
        _, _, data, _ = run(tmp_path, "v3_real_cut.json")
        ul, ur, ll, lr = data["camera"]["corners"][-1]
        top = (ur[0] - ul[0], ur[1] - ul[1])
        bottom = (lr[0] - ll[0], lr[1] - ll[1])
        assert abs(top[0] - bottom[0]) > 1.0 or abs(top[1] - bottom[1]) > 1.0

    @staticmethod
    def _block(monkeypatch, *prefixes):
        real_import = builtins.__import__

        def blocked(name, *a, **k):
            if name.startswith(prefixes):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocked)

    def _payload(self):
        data = json.loads((FIXTURES / "v3_real_cut.json").read_text())
        return conv.build_hierarchy_payload(
            str(FIXTURES / "v3_real_cut.json"), data, set(data["objects"]),
            {0: [1, 0, 0]}, 3840, 2160, 24.0, 1, 24)

    def test_vendored_copy_needs_nothing_installed(self, monkeypatch):
        self._block(monkeypatch, "rotobot_nuke")
        assert self._payload()["mode"] == "hierarchy"

    def test_missing_hierarchy_code_explains_the_fix(self, monkeypatch):
        self._block(monkeypatch, "rotobot_nuke", "_rotobot_hierarchy")
        with pytest.raises(SystemExit) as e:
            self._payload()
        assert "_rotobot_hierarchy" in str(e.value)
        assert "--flat" in str(e.value)


def drifting_camera(tmp_path, persp=0.0004):
    """v3_real_cut.json with a camera that runs away in perspective (as a tracker
    that has lost the plate does): frame k gets a horizontal projective term
    growing as persp * k, tagged as a good solve."""
    data = json.loads((FIXTURES / "v3_real_cut.json").read_text())
    for k, f in enumerate(sorted(data["camera"], key=int)):
        cam = data["camera"][f]
        cam["H2d"] = [1, 0, 0, 0, 1, 0, persp * k, 0, 1]
        cam["source"] = "ecc"
    src = tmp_path / "clip.json"
    src.write_text(json.dumps(data))
    return src


def run_src(src, *args):
    out = subprocess.run([sys.executable, str(SCRIPT), str(src), *args],
                         capture_output=True, text=True, cwd=src.parent)
    assert out.returncode == 0, out.stderr
    return src.with_suffix(".jsx"), json.loads((src.parent / "clip_data.json").read_text()), out.stdout


class TestCornerPinGuard:
    W, H = 3840, 2160
    PLATE = [[0, 0], [3840, 0], [0, 2160], [3840, 2160]]  # UL, UR, LL, LR

    def test_a_plain_plate_passes(self):
        assert conv.corner_pin_problem([self.PLATE] * 3, self.W, self.H) is None

    def test_a_folded_quad_is_caught(self):
        ul, ur, ll, lr = self.PLATE
        first, count, why = conv.corner_pin_problem([self.PLATE, [ul, ur, lr, ll]], self.W, self.H)
        assert (first, count) == (1, 1) and "folds" in why

    def test_an_oversized_quad_is_caught(self):
        wide = [[0, 0], [40000, 0], [0, 2160], [40000, 2160]]
        first, count, why = conv.corner_pin_problem([self.PLATE, wide, wide], self.W, self.H)
        assert (first, count) == (1, 2) and "across" in why

    def test_a_long_pan_keeps_its_camera(self):
        # A pan carries a wide precomp across the plate without enlarging it.
        pw, ph = 12000, 2160
        quads = [[[-x, 0], [pw - x, 0], [-x, ph], [pw - x, ph]] for x in range(0, 8000, 400)]
        assert conv.corner_pin_problem(quads, pw, ph) is None

    def test_magnifying_the_precomp_is_caught(self):
        big = [[0, 0], [3840 * 2.5, 0], [0, 2160], [3840, 2160]]
        assert conv.corner_pin_problem([self.PLATE, big], self.W, self.H) is not None

    def test_a_pin_that_explodes_later_is_caught_from_that_frame(self):
        # The shape seen on a lost track: a sane quad, then one that keeps widening.
        quads = [self.PLATE] * 5 + [[[0, 0], [3840 * k, 0], [0, 2160], [3840, 2160]] for k in (2, 6, 9)]
        first, count, why = conv.corner_pin_problem(quads, self.W, self.H)
        assert (first, count) == (6, 2) and "across" in why

    def test_a_real_camera_is_kept(self, tmp_path):
        _, _, data, stdout = run(tmp_path, "v3_real_cut.json")
        assert data["camera_dropped"] is None
        assert "NOTE: camera not usable" not in stdout

    def test_a_drifting_camera_is_dropped(self, tmp_path):
        _, data, stdout = run_src(drifting_camera(tmp_path))
        assert data["mode"] == "hierarchy"
        assert data["camera_dropped"].startswith("camera not usable in After Effects")
        assert data["held"][0] == data["camera_dropped"]
        assert "NOTE: camera not usable" in stdout
        # Without the camera the precomp is the plate plus its margins, and the
        # pin is the same plate-sized quad on every frame.
        pre = data["precomp"]
        assert pre["width"] <= 3840 + 4 * conv.PRECOMP_MARGIN
        assert pre["height"] <= 2160 + 4 * conv.PRECOMP_MARGIN
        assert conv.corner_pin_problem(data["camera"]["corners"], pre["width"], pre["height"]) is None
        assert all(c == data["camera"]["corners"][0] for c in data["camera"]["corners"])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestLoaderUnderMockAE:
    def test_loader_recomposes_every_vertex_without_the_camera(self, tmp_path):
        src = drifting_camera(tmp_path)
        jsx, data, _ = run_src(src)
        assert data["camera_dropped"]
        out = subprocess.run(["node", str(HERE / "ae_mock.js"), str(jsx), str(src)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        res = json.loads(out.stdout)
        assert res["shape_layers"] == 2 and res["parented"] == 2 and res["nulls"] == 1
        assert res["vertices_checked"] > 0
        assert res["worst_px"] < 0.05, res

    def test_loader_recomposes_every_vertex(self, tmp_path):
        src, jsx, _, _ = run(tmp_path, "v3_real_cut.json")
        out = subprocess.run(["node", str(HERE / "ae_mock.js"), str(jsx), str(src)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        res = json.loads(out.stdout)
        assert res["shape_layers"] == 2 and res["parented"] == 2 and res["nulls"] == 1
        assert res["vertices_checked"] > 0
        assert res["worst_px"] < 0.05, res


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestNoModalOnSuccess:
    """A success-path alert can open behind the host window on Windows, and
    the host then looks frozen. With the operator's own comp active at
    another rate, the fps note goes to the comp comment and the palette."""

    def _run(self, tmp_path, *args):
        src = tmp_path / "clip.json"
        doc = json.loads((FIXTURES / "v3_real_cut.json").read_text())
        doc["fps"], doc["frame_start"] = 30, 1001
        src.write_text(json.dumps(doc))
        r = subprocess.run([sys.executable, str(SCRIPT), str(src), *args],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = subprocess.run(["node", str(HERE / "ae_mock.js"), str(tmp_path / "clip.jsx"), str(src)],
                             capture_output=True, text=True,
                             env={**os.environ, "AE_MOCK_ACTIVE_FPS": "24"})
        assert out.returncode == 0, out.stderr      # alert() throws in the mock
        return json.loads(out.stdout)

    @pytest.mark.parametrize("args", [(), ("--flat",)])
    def test_fps_note_without_a_dialog(self, tmp_path, args):
        res = self._run(tmp_path, *args)
        assert res["comment"].startswith("artist notes\nTokgan import: This clip runs at 30 fps")
        (pal,) = res["palette"]
        assert pal["type"] == "palette" and pal["shown"]
        assert pal["text"][-1].startswith("Imported ") and "runs at 30 fps" in pal["text"][-1]
        assert "find-sidecar" in res["log"] and "] done: Imported " in res["log"]
        assert "note: This clip runs at 30 fps" in res["log"]

    def test_a_rerun_replaces_the_note(self, tmp_path):
        res = self._run(tmp_path, "--flat")
        assert res["comment"].count("Tokgan import:") == 1


class TestControlPointCap:
    def test_dense_keys_are_resampled_to_the_cap(self):
        import math
        pts = []
        for i in range(200):
            a = 2 * math.pi * i / 200
            x, y, tx, ty, h = 500 * math.cos(a), 500 * math.sin(a), -math.sin(a), math.cos(a), 5.236
            pts.append(dict(x=x, y=y, left_x=x - tx * h, left_y=y - ty * h,
                            right_x=x + tx * h, right_y=y + ty * h))
        out = conv.resample_points(pts, True, conv.DEFAULT_MAX_CP)
        assert conv.DEFAULT_MAX_CP == 96 and len(out) == 96
        assert all(abs(math.hypot(p["x"], p["y"]) - 500) < 1e-3 for p in out)
        assert conv.resample_points(pts, True, 0) is pts
        few = pts[:50]
        assert conv.resample_points(few, True, 96) is few


class TestConvertAPI:
    """convert() is what Rotobot Queue calls; the CLI is a wrapper over it."""

    def test_returns_what_it_wrote(self, tmp_path):
        src = tmp_path / "clip.json"
        shutil.copy(FIXTURES / "v3_real_cut.json", src)
        lines = []
        r = conv.convert(str(src), log=lines.append)
        assert r.mode == "hierarchy" and r.shapes == 2 and r.held == []
        assert Path(r.jsx).is_file() and Path(r.data).is_file()
        assert Path(r.data).name == "clip_data.json"
        assert any("Hierarchy:" in s for s in lines)

    def test_explicit_output_and_flat(self, tmp_path):
        src = tmp_path / "clip.json"
        shutil.copy(FIXTURES / "v3_real_cut.json", src)
        r = conv.convert(str(src), str(tmp_path / "shot_010.jsx"), flat=True,
                         log=lambda s: None)
        assert r.mode == "flat"
        assert Path(r.jsx).name == "shot_010.jsx"
        assert Path(r.data).name == "shot_010_data.json"
