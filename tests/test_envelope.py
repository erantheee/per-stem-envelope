import numpy as np
import pytest

from stemenvelope.envelope import PerStemEnvelope


def _content(n_stems=2, n_bands=8, peak_index=2):
    content = np.full((n_stems, n_bands), -120.0)   # -120 dB = numerically silent
    for stem in range(n_stems):
        content[stem, peak_index] = 10.0
    return content


def test_presence_peaks_at_content_and_fades():
    env = PerStemEnvelope(n_stems=1, n_bands=6, presence_window_db=24.0)
    content = np.array([[-120.0, -120.0, 10.0, -2.0, -14.0, -26.0]])
    presence = env.presence(content)
    assert presence.shape == (1, 6)
    assert np.isclose(presence[0, 2], 1.0)          # loudest band
    assert presence[0, 0] == 0.0                    # 130 dB down -> absent
    assert np.isclose(presence[0, 3], 0.5)          # 12 dB down of a 24 dB window
    assert presence[0, 5] == 0.0                    # 36 dB down -> absent


def test_silent_stem_gets_no_gain():
    env = PerStemEnvelope(n_stems=2, n_bands=4)
    silence = np.full((2, 4), -180.0)               # what band_db returns for zeros
    target = env.target(silence, np.full(4, 3.0))
    assert np.allclose(target, 0.0)


def test_empty_band_of_an_active_stem_gets_no_gain():
    env = PerStemEnvelope(n_stems=1, n_bands=3, presence_window_db=24.0)
    content = np.array([[10.0, -120.0, -120.0]])
    curve = np.full(3, 3.0)
    target = env.target(content, curve)
    assert np.isclose(target[0, 0], 3.0)
    assert target[0, 1] == 0.0 and target[0, 2] == 0.0


def test_target_is_bounded():
    env = PerStemEnvelope(n_stems=1, n_bands=2, max_boost_db=2.0, max_cut_db=-6.0)
    content = np.array([[10.0, 10.0]])
    assert np.allclose(env.target(content, np.array([50.0, -50.0])), [2.0, -6.0])


def test_step_reaches_63_percent_after_one_tau():
    env = PerStemEnvelope(n_stems=1, n_bands=1, tau_rise_s=1.0, tau_fall_s=1.0,
                          presence_window_db=24.0, max_boost_db=6.0)
    content = np.array([[10.0]])
    gains = env.step(content, np.array([6.0]), dt_s=1.0)
    assert np.isclose(gains[0, 0], 6.0 * (1 - np.exp(-1.0)), atol=1e-9)


def test_tau_controls_speed():
    rise = PerStemEnvelope(n_stems=1, n_bands=1, tau_rise_s=0.5, tau_fall_s=0.5)
    content = np.array([[10.0]])
    fast = rise.step(content, np.array([6.0]), dt_s=0.1)[0, 0]
    slow = PerStemEnvelope(n_stems=1, n_bands=1, tau_rise_s=1.0, tau_fall_s=1.0
                           ).step(content, np.array([6.0]), dt_s=0.1)[0, 0]
    assert fast > slow > 0.0


def test_inactive_tick_decays_to_zero():
    env = PerStemEnvelope(n_stems=1, n_bands=1, tau_rise_s=0.1, tau_fall_s=0.1,
                          presence_window_db=24.0)
    content = np.array([[10.0]])
    for _ in range(50):
        env.step(content, np.array([-6.0]), dt_s=0.05)
    assert env.gains_db[0, 0] < -5.0
    for _ in range(50):
        env.step(content, np.array([-6.0]), dt_s=0.05, active=False)
    assert abs(env.gains_db[0, 0]) < 1e-6


def test_reset_and_retarget():
    env = PerStemEnvelope(n_stems=4, n_bands=6)
    env.gains_db[:] = -3.0
    env.reset()
    assert np.allclose(env.gains_db, 0.0)
    env.reset(n_stems=6)
    assert env.gains_db.shape == (6, 6)


def test_layout_validation():
    env = PerStemEnvelope(n_stems=2, n_bands=4)
    with pytest.raises(ValueError):
        env.presence(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        env.target(np.zeros((2, 4)), np.zeros(5))
    with pytest.raises(ValueError):
        PerStemEnvelope(n_stems=0, n_bands=4)
    with pytest.raises(ValueError):
        PerStemEnvelope(n_stems=1, n_bands=4, presence_window_db=0.0)
    with pytest.raises(ValueError):
        PerStemEnvelope(n_stems=1, n_bands=4, max_boost_db=-1.0)


def test_peak_db_telemetry_is_per_stem():
    env = PerStemEnvelope(n_stems=2, n_bands=2, presence_window_db=24.0,
                          tau_rise_s=0.01, tau_fall_s=0.01)
    content = np.array([[10.0, -120.0], [-120.0, 10.0]])   # each stem owns one band
    for _ in range(200):
        env.step(content, np.array([-4.0, 0.0]), dt_s=0.01)
    peaks = env.peak_db()
    assert peaks.shape == (2,)
    assert np.allclose(peaks, [4.0, 0.0], atol=1e-3)
