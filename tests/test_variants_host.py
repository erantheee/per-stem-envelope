import json
import os
import time

import pytest

import tuning_bridge as tb


def test_modes_are_documented_and_bounded():
    for mode in tb.MODES:
        payload = tb.mode_payload(mode)
        assert 1.0 <= payload["depth"] <= 4.0
        assert 0.5 <= payload["masking_dose"] <= 2.0
        assert 1.0 <= payload["max_positive_db"] <= 6.0
    with pytest.raises(ValueError):
        tb.mode_payload("loud")


def test_off_is_a_true_bypass():
    off = tb.mode_payload("off")
    assert off["depth"] == 1.0 and off["max_positive_db"] == 1.0


def test_set_mode_writes_control_file(tmp_path):
    control = tmp_path / "tuning-control.json"
    bridge = tb.TuningBridge(control_path=control, telemetry_path=tmp_path / "none.json")
    payload = bridge.set_mode("aggressive")
    assert json.loads(control.read_text()) == payload
    assert bridge.mode == "aggressive"
    assert not list(tmp_path.glob("*.writing"))


def test_telemetry_missing_returns_none(tmp_path):
    bridge = tb.TuningBridge(control_path=tmp_path / "c.json",
                             telemetry_path=tmp_path / "absent.json")
    assert bridge.telemetry() is None
    assert bridge.summary()["stale"] is True


def test_telemetry_stale_returns_none(tmp_path):
    telemetry = tmp_path / "tuning-latest.json"
    telemetry.write_text(json.dumps({"mode": "x"}), encoding="utf-8")
    old = time.time() - 60
    os.utime(telemetry, (old, old))
    bridge = tb.TuningBridge(control_path=tmp_path / "c.json", telemetry_path=telemetry)
    assert bridge.telemetry() is None


def test_telemetry_garbage_returns_none(tmp_path):
    telemetry = tmp_path / "tuning-latest.json"
    telemetry.write_text("{not json", encoding="utf-8")
    bridge = tb.TuningBridge(control_path=tmp_path / "c.json", telemetry_path=telemetry)
    assert bridge.telemetry() is None


def test_summary_reports_peak_and_stems(tmp_path):
    telemetry = tmp_path / "tuning-latest.json"
    telemetry.write_text(json.dumps({
        "mode": "PER_STEM_ENVELOPE",
        "stale": False,
        "stem_db": [[0.0, -1.5], [2.0, 0.0]],
        "stem_peak_db": [1.5, 2.0],
        "control": {"depth": 1.3},
    }), encoding="utf-8")
    bridge = tb.TuningBridge(control_path=tmp_path / "c.json", telemetry_path=telemetry)
    summary = bridge.summary()
    assert summary["mode"] == "PER_STEM_ENVELOPE"
    assert summary["peak_db"] == 2.0
    assert summary["stems"]["0"] == [0.0, -1.5]
    assert summary["control"]["depth"] == 1.3


def test_http_path_tolerates_unreachable_url(tmp_path):
    bridge = tb.TuningBridge(control_path=tmp_path / "c.json",
                             telemetry_path=tmp_path / "t.json",
                             telemetry_url="http://127.0.0.1:9/v1/tuning",
                             timeout_s=0.05)
    assert bridge.telemetry() is None
