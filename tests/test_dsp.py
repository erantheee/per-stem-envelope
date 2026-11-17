import numpy as np
import pytest

from stemenvelope.dsp import (apply_band_gains, frame_count, istft, render_stems,
                              stem_band_db, stft)
from stemenvelope.bands import BandMap

N_FFT = 512
HOP = 128


def _random_signal(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.1


def test_frame_count_matches_stft():
    for n in (100, N_FFT, N_FFT + 1, 6000):
        assert frame_count(n, N_FFT, HOP) == stft(_random_signal(n), N_FFT, HOP).shape[0]


def test_stft_istft_roundtrip_interior():
    x = _random_signal()
    rec = istft(stft(x, N_FFT, HOP), N_FFT, HOP, length=x.size)
    interior = slice(N_FFT, x.size - N_FFT)
    assert np.max(np.abs(rec[interior] - x[interior])) < 1e-9


def test_apply_band_gains_zero_is_identity():
    spectrum = stft(_random_signal(), N_FFT, HOP)
    band_map = BandMap(n_fft=N_FFT)
    out = apply_band_gains(spectrum, np.zeros(band_map.n_bands), band_map)
    assert np.allclose(out, spectrum)


def test_apply_band_gains_constant_gain_is_scalar():
    spectrum = stft(_random_signal(), N_FFT, HOP)
    band_map = BandMap(n_fft=N_FFT)
    out = apply_band_gains(spectrum, np.full(band_map.n_bands, -6.0), band_map)
    expected = spectrum * 10 ** (-6.0 / 20.0)
    assert np.allclose(out, expected)


def test_apply_band_gains_per_frame():
    spectrum = stft(_random_signal(), N_FFT, HOP)
    band_map = BandMap(n_fft=N_FFT)
    gains = np.zeros((spectrum.shape[0], band_map.n_bands))
    gains[:, :] = 0.0
    gains[0, :] = -12.0
    out = apply_band_gains(spectrum, gains, band_map)
    assert np.allclose(out[1:], spectrum[1:])
    assert np.allclose(out[0], spectrum[0] * 10 ** (-12.0 / 20.0))


def test_apply_band_gains_frame_mismatch():
    band_map = BandMap(n_fft=N_FFT)
    spectrum = stft(_random_signal(), N_FFT, HOP)
    with pytest.raises(ValueError):
        apply_band_gains(spectrum, np.zeros((spectrum.shape[0] + 1, 24)), band_map)


def _two_stems(n=6000, sr=44100):
    t = np.arange(n) / sr
    low = 0.2 * np.sin(2 * np.pi * 120 * t)
    high = 0.2 * np.sin(2 * np.pi * 5000 * t)
    return np.stack([low, high])


def test_render_zero_gains_sums_stems():
    stems = _two_stems()
    band_map = BandMap(n_fft=N_FFT)
    mix = render_stems(stems, np.zeros((2, band_map.n_bands)), band_map, N_FFT, HOP)
    expected = stems.sum(axis=0)
    interior = slice(N_FFT, stems.shape[1] - N_FFT)
    assert np.max(np.abs(mix[interior] - expected[interior])) < 1e-9


def test_render_cut_hits_only_target_stem_band():
    stems = _two_stems()
    band_map = BandMap(n_fft=N_FFT)
    gains = np.zeros((2, band_map.n_bands))
    gains[0, band_map.band_index(120.0)] = -30.0
    mix = render_stems(stems, gains, band_map, N_FFT, HOP)
    # removing the low stem must drop low-frequency energy a lot, keep high intact
    low_band = band_map.band_index(120.0)
    high_band = band_map.band_index(5000.0)
    spec_before = np.abs(stft(stems.sum(axis=0), N_FFT, HOP)) ** 2
    spec_after = np.abs(stft(mix, N_FFT, HOP)) ** 2
    def band_energy(spec):
        power = band_map.band_power(spec.T).mean(axis=1)
        return 10.0 * np.log10(np.maximum(power, 1e-18))

    drop_low = band_energy(spec_after)[low_band] - band_energy(spec_before)[low_band]
    drop_high = band_energy(spec_after)[high_band] - band_energy(spec_before)[high_band]
    assert drop_low < -6.0
    assert abs(drop_high) < 1.5


def test_stem_band_db_shape():
    stems = _two_stems()
    n_fft = 2048          # 24 log bands need real resolution at the low end
    band_map = BandMap(n_fft=n_fft)
    levels = stem_band_db(stems, band_map, n_fft, HOP)
    assert levels.shape == (2, stem_band_db(stems[:1], band_map, n_fft, HOP).shape[1], 24)
    assert int(np.argmax(levels[0, 3])) == band_map.band_index(120.0)
    assert int(np.argmax(levels[1, 3])) == band_map.band_index(5000.0)
