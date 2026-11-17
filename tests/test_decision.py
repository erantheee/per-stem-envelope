import pytest

from stemenvelope.decision import (ENVIRONMENTS, EnvironmentTracker, apply_environment,
                                   noise_exposure)


def test_environments_are_the_documented_labels():
    assert ENVIRONMENTS == ("quiet", "speech", "babble", "street", "office", "wind", "keyboard")


def test_unknown_evidence_fails_open_to_quiet():
    tracker = EnvironmentTracker()
    tracker.update("street")
    assert tracker.state == "street"
    assert tracker.update("unknown-label") == "quiet"
    assert tracker.update(None) == "quiet"


def test_first_label_latches_immediately():
    tracker = EnvironmentTracker()
    assert tracker.update("office") == "office"


def test_hysteresis_needs_consecutive_windows():
    tracker = EnvironmentTracker(hold_windows=3)
    tracker.update("quiet")
    tracker.update("street")
    tracker.update("quiet")          # interruption resets the counter
    tracker.update("street")
    tracker.update("street")
    assert tracker.state == "quiet"
    assert tracker.update("street") == "street"


def test_speech_enters_faster_than_other_states():
    tracker = EnvironmentTracker(hold_windows=3, fast_windows=2)
    tracker.update("office")
    tracker.update("speech")
    assert tracker.update("speech") == "speech"


def test_speech_release_still_needs_full_hold():
    tracker = EnvironmentTracker(hold_windows=3, fast_windows=2)
    tracker.update("speech")
    tracker.update("office")
    tracker.update("office")
    assert tracker.state == "speech"
    assert tracker.update("office") == "office"


def test_noise_exposure_mapping():
    assert noise_exposure(None) == 1.0            # fail open
    assert noise_exposure("not-a-number") == 1.0
    assert noise_exposure(-90.0) == 0.0
    assert noise_exposure(-46.0) == 1.0
    assert noise_exposure(-40.0) == 1.0
    assert abs(noise_exposure(-56.0) - 0.5) < 1e-9


def test_apply_environment_keyboard_table():
    gains = apply_environment({}, "keyboard")
    assert gains["drums"] == pytest.approx(1.5)
    assert gains["other"] == pytest.approx(0.3)


def test_apply_environment_quiet_is_identity():
    assert apply_environment({}, "quiet") == {}


def test_apply_environment_scales_with_noise():
    high = apply_environment({}, "keyboard", env_scale=1.0, noise_scale=1.0)["drums"]
    low = apply_environment({}, "keyboard", env_scale=1.0, noise_scale=0.0)["drums"]
    assert high > low                      # NOISE_BASE_FACTOR keeps it non-zero
    assert low == pytest.approx(1.5 * 0.7)


def test_apply_environment_speech_ignores_gear_and_level():
    a = apply_environment({}, "speech", env_scale=0.2, noise_scale=0.0)["vocals"]
    b = apply_environment({}, "speech", env_scale=3.0, noise_scale=1.0)["vocals"]
    assert a == b == pytest.approx(1.0)


def test_apply_environment_floor_and_ceiling():
    gains = apply_environment({"vocals": -100.0, "drums": 100.0}, "quiet")
    assert gains["vocals"] == pytest.approx(-9.0)
    assert gains["drums"] == pytest.approx(1.5)


def test_apply_environment_custom_table():
    table = {"wind": {"other": 3.0}}
    assert apply_environment({}, "wind", modifiers=table)["other"] == pytest.approx(1.5)
