"""Enforce DATA_POLICY.md: no audio, no model weights, ever.

This is not paranoia.  The whole point of keeping the algorithm separate from the
data is that the algorithm can be public while the data cannot; a single
`git add song.wav` silently reverses that.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    # audio
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".wem",
    # model weights
    ".onnx", ".rknn", ".pt", ".pth", ".ckpt", ".safetensors", ".tflite", ".h5",
}


def _tracked_files() -> list:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    return [p for p in result.stdout.decode("utf-8", "replace").split("\0") if p]


def test_no_audio_or_weights_are_tracked():
    tracked = _tracked_files()
    assert tracked, "expected at least one tracked file"
    offending = [p for p in tracked if Path(p).suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not offending, (
        "DATA_POLICY violation — audio/weights must not be committed: "
        f"{offending}. See DATA_POLICY.md.")


def test_gitignore_covers_media_and_weights():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for suffix in (".wav", ".onnx", ".rknn", ".pt"):
        assert suffix in ignore, f".gitignore must list {suffix}"
