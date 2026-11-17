"""Enforce DATA_POLICY.md.

Two rules, both mechanical:

1. **No audio, ever.**  The algorithm is separable from the data, and that is the
   only reason it can be public.  A single ``git add song.wav`` reverses that.
2. **Model weights only under ``weights/``**, and only when declared in
   ``weights/MANIFEST.json`` with a licence tag.  Weights carry their own licence
   (some research-only); they must never be draggable into the MIT part by
   accident.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".wem"}
BINARY_SUFFIXES = {".onnx", ".rknn", ".pt", ".pth", ".ckpt", ".safetensors", ".tflite", ".h5"}
WEIGHTS_PREFIX = "weights/"


def _tracked_files() -> list:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    return [p for p in result.stdout.decode("utf-8", "replace").split("\0") if p]


def _declared_weights() -> set:
    manifest = json.loads((ROOT / "weights/MANIFEST.json").read_text(encoding="utf-8"))
    return {Path(a["path"]).name for a in manifest["artifacts"]}


def test_no_audio_is_tracked():
    tracked = _tracked_files()
    assert tracked, "expected at least one tracked file"
    offending = [p for p in tracked if Path(p).suffix.lower() in AUDIO_SUFFIXES]
    assert not offending, (
        "DATA_POLICY violation — audio must never be committed: "
        f"{offending}. See DATA_POLICY.md.")


def test_weights_live_only_in_weights_dir():
    offending = [p for p in _tracked_files()
                 if Path(p).suffix.lower() in BINARY_SUFFIXES
                 and not p.startswith(WEIGHTS_PREFIX)]
    assert not offending, (
        "model binaries must live under weights/ so their licence stays separated: "
        f"{offending}")


def test_tracked_weights_are_declared_in_manifest():
    declared = _declared_weights()
    tracked = [Path(p).name for p in _tracked_files()
               if p.startswith(WEIGHTS_PREFIX) and Path(p).suffix.lower() in BINARY_SUFFIXES]
    undeclared = [name for name in tracked if name not in declared]
    assert not undeclared, f"weights committed without a MANIFEST entry: {undeclared}"


def test_gitignore_covers_media_and_weights():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for suffix in (".wav", ".onnx", ".rknn", ".pt"):
        assert suffix in ignore, f".gitignore must list {suffix}"
