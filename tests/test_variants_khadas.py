import json
import time
from pathlib import Path

import numpy as np
import pytest

import board_adapter as ba
from stemenvelope import BandMap

N_BANDS = 24


def _payload(**overrides):
    payload = {
        "level_dbfs": -54.2,
        "band_dbfs": [-80.0] * N_BANDS,
        "event": "keyboard",
        "voice": 0.1,
        "conversation_cue": 0,
        "ts_ms": int(time.time() * 1000),
    }
    payload.update(overrides)
    return payload


def test_evidence_translation():
    evidence = ba.evidence_from_board(_payload(), status={"policy_state": "IDLE"})
    assert evidence.level_dbfs == -54.2
    assert evidence.event == "keyboard"
    assert evidence.valid_bands(N_BANDS) is not None
    assert evidence.valid_conversation() == "IDLE"


def test_conversation_cue_overrides_missing_status():
    evidence = ba.evidence_from_board(_payload(conversation_cue=1))
    assert evidence.valid_conversation() == "ENGAGED"


def test_stale_payload_fails_open():
    old = _payload(ts_ms=int((time.time() - 30) * 1000))
    evidence = ba.evidence_from_board(old, max_age_s=2.5)
    assert evidence.valid_level() is None, "stale evidence must not drive gains"


def test_garbage_payload_fails_open():
    for bad in (None, [], "nope", 42):
        assert ba.evidence_from_board(bad).valid_level() is None
    assert ba.evidence_from_board({}).valid_level() is None
    assert ba.evidence_from_board(_payload(level_dbfs=float("nan"))).valid_level() is None
    assert ba.evidence_from_board(_payload(band_dbfs=[0.0] * 7)).valid_bands(N_BANDS) is None


def test_read_evidence_file(tmp_path):
    path = tmp_path / "dmic-latest.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert ba.read_evidence_file(path).valid_level() == -54.2
    # missing file must not raise
    assert ba.read_evidence_file(tmp_path / "absent.json").valid_level() is None


def test_control_payloads_are_in_worker_range():
    for mode in ("off", "basic", "aggressive"):
        payload = ba.control_payload(mode)
        for key, value in payload.items():
            low, high = ba.CONTROL_RANGES[key]
            assert low <= value <= high, f"{mode}.{key}={value} outside {ba.CONTROL_RANGES[key]}"
    with pytest.raises(ValueError):
        ba.control_payload("loud")


def test_clamp_control_matches_worker_clamps():
    clamped = ba.clamp_control({"depth": 99, "masking_dose": 0.1, "max_positive_db": -5,
                                "unknown": 1})
    assert clamped == {"depth": 4.0, "masking_dose": 0.5, "max_positive_db": 1.0}
    assert ba.clamp_control(None) == {}
    assert ba.clamp_control({"depth": True}) == {}     # bool is not a number here


class _Result:
    def __init__(self, gains, curve):
        self.gains_db = gains
        self.curve_db = curve


def test_telemetry_shape_matches_board_keys():
    rng = np.random.default_rng(0)
    gains = rng.normal(0.0, 1.0, size=(6, 10, N_BANDS))
    curve = rng.normal(0.0, 1.0, size=(10, N_BANDS))
    band_map = BandMap(n_fft=2048)
    payload = ba.telemetry_from_result(_Result(gains, curve), band_map,
                                       modes=ba.control_payload("basic"))
    for key in ("ts_ms", "mode", "style", "stale", "band_db", "stem_peak_db",
                "stem_db", "control", "band_centers_hz"):
        assert key in payload
    assert len(payload["stem_db"]) == 6
    assert len(payload["stem_db"][0]) == N_BANDS
    assert len(payload["stem_peak_db"]) == 6
    assert payload["control"]["depth"] == 1.3


def test_telemetry_rejects_bad_shape():
    band_map = BandMap(n_fft=2048)
    with pytest.raises(ValueError):
        ba.telemetry_from_result(_Result(np.zeros((4, 24)), np.zeros(24)), band_map)


def test_write_telemetry_is_atomic(tmp_path):
    target = tmp_path / "tuning-latest.json"
    ba.write_telemetry(target, {"mode": "x"})
    assert json.loads(target.read_text())["mode"] == "x"
    assert not list(tmp_path.glob("*.writing")), "temporary file must not be left behind"
