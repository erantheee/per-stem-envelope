"""Host (上位机) side: panel gear in, telemetry out.

The host's only jobs are (a) let a human pick a gear, (b) show what the board is
actually doing.  It must never block the audio path, so telemetry is read from a
file the board rewrites (or a non-blocking HTTP GET) and parsed defensively.

Files, matching the board's conventions:

* ``/tmp/tuning-control.json``  host -> board   ``{depth, masking_dose, max_positive_db}``
* ``/tmp/tuning-latest.json``   board -> host   per-stem gains, curve, mode
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

DEFAULT_CONTROL_PATH = "/tmp/tuning-control.json"
DEFAULT_TELEMETRY_PATH = "/tmp/tuning-latest.json"

#: Panel gears.  ``off`` is a true bypass (depth 1.0 + zero positive headroom is
#: the board's own convention for "do not touch the audio").
MODES = {
    "off": {"depth": 1.0, "masking_dose": 1.0, "max_positive_db": 1.0},
    "basic": {"depth": 1.3, "masking_dose": 1.0, "max_positive_db": 3.0},
    "aggressive": {"depth": 2.5, "masking_dose": 1.3, "max_positive_db": 4.0},
}

#: Telemetry older than this is reported as stale (the board shows it too).
TELEMETRY_MAX_AGE_S = 3.0


def mode_payload(mode: str) -> dict:
    key = str(mode).lower()
    if key not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    return dict(MODES[key])


class TuningBridge:
    """Bidirectional host<->board tuning link, file based."""

    def __init__(self, control_path=DEFAULT_CONTROL_PATH,
                 telemetry_path=DEFAULT_TELEMETRY_PATH,
                 telemetry_url: str = None, timeout_s: float = 0.4) -> None:
        self.control_path = Path(control_path)
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_url = telemetry_url
        self.timeout_s = float(timeout_s)
        self._mode = None

    # ── host -> board ────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> dict:
        """Write the control file.  Returns what was written."""
        payload = mode_payload(mode)
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.control_path.with_suffix(self.control_path.suffix + ".writing")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.control_path)
        self._mode = str(mode).lower()
        return payload

    @property
    def mode(self):
        return self._mode

    # ── board -> host ────────────────────────────────────────────────────────
    def telemetry(self, max_age_s: float = TELEMETRY_MAX_AGE_S):
        """Latest telemetry dict, or ``None`` when missing/stale/unparseable."""
        if self.telemetry_url:
            return self._telemetry_http(max_age_s)
        try:
            age = time.time() - self.telemetry_path.stat().st_mtime
        except OSError:
            return None
        if age > float(max_age_s):
            return None
        try:
            payload = json.loads(self.telemetry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _telemetry_http(self, max_age_s: float):
        try:
            with urllib.request.urlopen(self.telemetry_url, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def summary(self) -> dict:
        """Compact, human-facing view for a panel/debug line."""
        payload = self.telemetry()
        if payload is None:
            return {"mode": self._mode, "stale": True, "stems": {}, "peak_db": 0.0}
        stem_db = payload.get("stem_db") or []
        peaks = payload.get("stem_peak_db") or [0.0] * len(stem_db)
        return {
            "mode": payload.get("mode", self._mode),
            "stale": bool(payload.get("stale", False)),
            "peak_db": round(max((abs(float(v)) for v in peaks), default=0.0), 2),
            "stems": {str(i): [round(float(v), 2) for v in row] for i, row in enumerate(stem_db)},
            "control": payload.get("control", {}),
        }
