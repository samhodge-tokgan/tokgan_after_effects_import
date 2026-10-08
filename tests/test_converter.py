"""Converter tests: flat and hierarchy payloads, and the hierarchy loader run
end to end under a mock of the After Effects scripting DOM (tests/ae_mock.js).

``v3_real_cut.json`` is cut from a real Rotobot Next 0.10.0 run: two body
parts over 24 4K frames with the real (projective) camera and pelvis track.
``v2_cut.json`` is the same data without the camera and persons blocks.
"""

from __future__ import annotations

import builtins
import json
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
        assert jsx.read_text() == conv.LOADER_TEMPLATE.format(data_basename="clip_data.json")

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


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestLoaderUnderMockAE:
    def test_loader_recomposes_every_vertex(self, tmp_path):
        src, jsx, _, _ = run(tmp_path, "v3_real_cut.json")
        out = subprocess.run(["node", str(HERE / "ae_mock.js"), str(jsx), str(src)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        res = json.loads(out.stdout)
        assert res["shape_layers"] == 2 and res["parented"] == 2 and res["nulls"] == 1
        assert res["vertices_checked"] > 0
        assert res["worst_px"] < 0.05, res


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
