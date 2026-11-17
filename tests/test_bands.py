import numpy as np
import pytest

from stemenvelope.bands import BandMap, DEFAULT_CENTERS_HZ


def _tone(hz, sr=44100, n=8192, amp=0.3):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * hz * t)


def test_default_layout():
    band_map = BandMap()
    assert band_map.n_bands == 24
    assert band_map.n_bins == 2048 // 2 + 1
    assert band_map.centers_hz is DEFAULT_CENTERS_HZ


def test_custom_layout_and_validation():
    band_map = BandMap(centers_hz=[100, 400, 1600, 6400], n_fft=1024)
    assert band_map.n_bands == 4
    with pytest.raises(ValueError):
        BandMap(centers_hz=[400, 100])
    with pytest.raises(ValueError):
        BandMap(centers_hz=[100])


def test_tone_lands_in_nearest_band():
    band_map = BandMap(n_fft=2048)
    for hz in (100.0, 1000.0, 4000.0, 10000.0):
        spectrum = np.fft.rfft(_tone(hz, n=2048) * np.hanning(2048))
        levels = band_map.band_db(np.abs(spectrum) ** 2)
        assert int(np.argmax(levels)) == band_map.band_index(hz), hz


def test_gains_to_bins_constant_gain():
    band_map = BandMap(n_fft=512)
    bins = band_map.gains_to_bins(np.full(24, -6.0))
    assert bins.shape == (band_map.n_bins,)
    assert np.allclose(bins, -6.0)


def test_gains_to_bins_interpolates_near_centre():
    band_map = BandMap(n_fft=2048)
    gains = np.zeros(24)
    index = band_map.band_index(1000.0)
    gains[index] = 6.0
    bins = band_map.gains_to_bins(gains)
    assert 4.0 < bins.max() <= 6.0      # no bin sits exactly on the centre
    freqs = np.fft.rfftfreq(2048, 1 / 44100)
    at_centre = np.abs(np.log2(np.maximum(freqs, 1e-3))
                            - np.log2(band_map.centers_hz[index])).argmin()
    assert bins[at_centre] > 4.0 and bins[at_centre] <= 6.0
    # far away from the inflated band the interpolation must not leak
    assert abs(bins[band_map.band_index(6000.0)]) < 0.5


def test_band_power_shapes_and_monotonicity():
    band_map = BandMap(n_fft=512)
    power = np.ones(band_map.n_bins)
    assert band_map.band_power(power).shape == (24,)
    frames = np.stack([power, power * 4.0], axis=1)
    levels = band_map.band_db(frames)
    assert levels.shape == (24, 2)
    assert np.allclose(levels[:, 1] - levels[:, 0], 10 * np.log10(4.0))


def test_band_power_rejects_wrong_width():
    band_map = BandMap(n_fft=512)
    with pytest.raises(ValueError):
        band_map.band_power(np.ones(7))
    with pytest.raises(ValueError):
        band_map.gains_to_bins(np.zeros(7))
