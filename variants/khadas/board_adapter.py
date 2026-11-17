"""Khadas (Edge2 / RKNPU) adapter: board payloads <-> this package's types.

The board already speaks JSON: evidence in, tuning out.  This module is the whole
seam, so the algorithm never learns about HTTP, ``/tmp`` files or the worker's
policy state machine.

Board evidence payload (``/v1/evidence``, ``/tmp/dmic-latest.json``)::

    {"level_dbfs": -54.2, "band_dbfs": [24 floats], "event": "keyboard",
     "voice": 0.1, "conversation_cue": 0}

Board tuning payload (``/tmp/tuning-latest.json``) keys are reproduced by
:func:`telemetry_from_result`; the host reads that file for its panel.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from stemenvelope import BandMap, Evidence

#: Panel gears the host can push to the board (``/tmp/tuning-control.json``).
MODE_CONTROLS = {
    "off": {"depth": 1.0, "masking_dose": 1.0, "max_positive_db": 1.0},
    "basic": {"depth": 1.3, "masking_dose": 1.0, "max_positive_db": 3.0},
    "aggressive": {"depth": 2.5, "masking_dose": 1.3, "max_positive_db": 4.0},
}

#: The worker clamps these on its side too; keep the two ranges identical.
CONTROL_RANGES = {"depth": (1.0, 4.0), "masking_dose": (0.5, 2.0),
                  "max_positive_db": (1.0, 6.0)}


def evidence_from_board(payload, status=None, max_age_s=None) -> Evidence:
    """Translate one board payload into :class:`Evidence`.

    ``status`` may carry ``policy_state`` (the conversation state).  A payload
    older than ``max_age_s`` (if it carries ``ts_ms``) is treated as missing
    evidence, i.e. it fails open.
    """
    if not isinstance(payload, dict):
        return Evidence(level_dbfs=None)
    if max_age_s is not None:
        ts_ms = payload.get("ts_ms")
        if isinstance(ts_ms, (int, float)) and not isinstance(ts_ms, bool):
            if (time.time() * 1000.0) - float(ts_ms) > float(max_age_s) * 1000.0:
                return Evidence(level_dbfs=None)
    conversation = None
    if isinstance(status, dict):
        conversation = status.get("policy_state")
    if not conversation and payload.get("conversation_cue"):
        conversation = "ENGAGED"
    return Evidence(
        level_dbfs=payload.get("level_dbfs", payload.get("level")),
        band_dbfs=payload.get("band_dbfs", payload.get("bands")),
        event=payload.get("event"),
        voice=payload.get("voice", 0.0),
        conversation=conversation,
    )


def read_evidence_file(path="/tmp/dmic-latest.json", max_age_s: float = 2.5) -> Evidence:
    """Read the board's evidence file with a staleness gate (fail-open)."""
    try:
        stale = time.time() - os.path.getmtime(path) > float(max_age_s)
    except OSError:
        return Evidence(level_dbfs=None)
    if stale:
        return Evidence(level_dbfs=None)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Evidence(level_dbfs=None)
    return evidence_from_board(payload)


def control_payload(mode: str) -> dict:
    """Panel gear -> the JSON the worker's control loop reads."""
    key = str(mode).lower()
    if key not in MODE_CONTROLS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODE_CONTROLS)}")
    return dict(MODE_CONTROLS[key])


def clamp_control(payload: dict) -> dict:
    """Clamp a control payload into the worker's accepted range."""
    out = {}
    if not isinstance(payload, dict):
        return out
    for key, (low, high) in CONTROL_RANGES.items():
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = max(low, min(high, float(value)))
    return out


def telemetry_from_result(result, band_map: BandMap, modes=None) -> dict:
    """Render pipeline output into the board's ``tuning-latest.json`` shape."""
    gains = np.asarray(result.gains_db, dtype=np.float64)
    if gains.ndim != 3:
        raise ValueError("expected gains_db [stems, frames, bands]")
    final = gains[:, -1, :]
    return {
        "ts_ms": int(time.time() * 1000),
        "mode": str(getattr(result, "mode", "PER_STEM_ENVELOPE")),
        "style": "per_stem",
        "stale": False,
        "band_db": [round(float(v), 2) for v in np.asarray(result.curve_db)[-1]],
        "stem_peak_db": [round(float(v), 2) for v in np.max(np.abs(gains), axis=(1, 2))],
        "stem_db": [[round(float(v), 2) for v in row] for row in final],
        "control": dict(modes or {}),
        "band_centers_hz": [float(v) for v in band_map.centers_hz],
    }


def write_telemetry(path, payload) -> None:
    """Atomic-ish write so the host never reads a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".writing")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
