"""Contract tests for the weights directory.

Code is MIT; weights are not.  This file is what keeps the two from being
confused: every committed binary must be declared in MANIFEST.json, hash-match
it, and carry a licence tag.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
MANIFEST = WEIGHTS / "MANIFEST.json"

BINARY_SUFFIXES = {".pt", ".onnx", ".rknn", ".pth", ".ckpt", ".safetensors"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_parses_and_has_artifacts():
    payload = _manifest()
    assert payload["artifacts"], "MANIFEST.json must list the artifacts"
    for artifact in payload["artifacts"]:
        assert artifact["path"].startswith("weights/")
        assert (ROOT / artifact["path"]).is_file(), f"missing file {artifact['path']}"
        assert len(artifact["sha256"]) == 64


def test_every_artifact_hash_matches():
    for artifact in _manifest()["artifacts"]:
        actual = _sha256(ROOT / artifact["path"])
        assert actual == artifact["sha256"], (
            f"{artifact['path']} hash drift: manifest says {artifact['sha256']}, file is {actual}. "
            "If the model really changed, update MANIFEST.json and LICENSE-WEIGHTS.md together.")


def test_every_binary_is_declared():
    declared = {Path(a["path"]).name for a in _manifest()["artifacts"]
                if Path(a["path"]).suffix in BINARY_SUFFIXES}
    on_disk = {p.name for p in WEIGHTS.iterdir() if p.suffix in BINARY_SUFFIXES}
    assert on_disk == declared, f"undeclared weights: {on_disk - declared}"


def test_each_binary_states_its_licence_and_commercial_use():
    for artifact in _manifest()["artifacts"]:
        if Path(artifact["path"]).suffix not in BINARY_SUFFIXES:
            continue
        assert artifact.get("license"), f"{artifact['path']} has no licence tag"
        assert "commercial_use" in artifact, f"{artifact['path']} has no commercial_use flag"
        assert artifact.get("training_data"), f"{artifact['path']} does not state its training data"


def test_no_noncommercial_weight_claims_mit():
    for artifact in _manifest()["artifacts"]:
        if artifact.get("commercial_use") is False:
            assert "NC" in artifact["license"].upper(), (
                f"{artifact['path']} is not commercial-use but its licence says "
                f"{artifact['license']!r}")


def test_weights_licence_file_exists_and_names_the_split():
    text = (WEIGHTS / "LICENSE-WEIGHTS.md").read_text(encoding="utf-8").lower()
    assert "mit" in text
    assert "nc" in text
    assert "musdb" in text
