"""STFT helpers and per-band gain application.

Deliberately minimal and dependency-free (numpy only).  If you already run your
own STFT pipeline, skip this module and call the envelope directly — the
envelope's output is just ``[n_stems, n_bands]`` dB at a known tick rate.
"""
from __future__ import annotations

import numpy as np

from .bands import BandMap

DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512


def _window(n_fft: int) -> np.ndarray:
    return np.hanning(int(n_fft))


def frame_count(n_samples: int, n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP) -> int:
    """Number of frames :func:`stft` will produce for ``n_samples``.

    Ceil, not floor: the last partial hop must still be covered, otherwise the
    reconstruction would be shorter than the input and callers would silently
    lose the tail.
    """
    n_fft, hop = int(n_fft), int(hop)
    if n_samples < n_fft:
        return 1
    return 1 + -(-(int(n_samples) - n_fft) // hop)


def stft(signal, n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP) -> np.ndarray:
    """Hann STFT -> complex ``[frames, bins]``, zero-padded to whole frames."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n_fft, hop = int(n_fft), int(hop)
    if hop < 1 or n_fft < 2 or hop > n_fft:
        raise ValueError("need 2 <= n_fft and 1 <= hop <= n_fft")
    n_frames = frame_count(x.size, n_fft, hop)
    pad = max(0, (n_frames - 1) * hop + n_fft - x.size)
    padded = np.concatenate([x, np.zeros(pad)])
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(padded[idx] * _window(n_fft)[None, :], axis=1)


def istft(spectrum, n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP,
          length=None) -> np.ndarray:
    """Weighted overlap-add inverse of :func:`stft`.

    The first and last ``hop`` samples are attenuated (a Hann frame starts at
    zero), which is inherent to WOLA without pre-padding; callers that need the
    edges exact should pad the input.
    """
    spec = np.asarray(spectrum)
    n_fft, hop = int(n_fft), int(hop)
    if spec.ndim != 2 or spec.shape[1] != n_fft // 2 + 1:
        raise ValueError(f"expected [frames, {n_fft // 2 + 1}], got {spec.shape}")
    n_frames = spec.shape[0]
    win = _window(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win[None, :]
    total = (n_frames - 1) * hop + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    win_sq = win ** 2
    for i in range(n_frames):
        start = i * hop
        out[start:start + n_fft] += frames[i]
        norm[start:start + n_fft] += win_sq
    out /= np.maximum(norm, 1e-8)
    return out[:length] if length is not None else out


def apply_band_gains(spectrum, gains_db, band_map: BandMap) -> np.ndarray:
    """Multiply a complex spectrum by per-band dB gains.

    ``gains_db`` may be one curve ``[bands]`` (static) or one per frame
    ``[frames, bands]``.
    """
    spec = np.asarray(spectrum)
    gains = np.asarray(gains_db, dtype=np.float64)
    if gains.ndim == 1:
        gains = np.broadcast_to(gains, (spec.shape[0], gains.shape[0]))
    if gains.shape[0] != spec.shape[0]:
        raise ValueError(f"gains have {gains.shape[0]} frames, spectrum has {spec.shape[0]}")
    out = np.empty(spec.shape, dtype=np.complex128)
    for frame in range(spec.shape[0]):
        linear = 10.0 ** (band_map.gains_to_bins(gains[frame]) / 20.0)
        out[frame] = spec[frame] * linear
    return out


def stem_band_db(stems, band_map: BandMap,
                 n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP) -> np.ndarray:
    """Per-stem band levels ``[n_stems, frames, bands]`` in dB.

    These are the "content features" the envelope gates on.  They come from the
    stems themselves, not from the separator's masks, so this works with any
    separator (or any set of real multitrack files).
    """
    audio = np.atleast_2d(np.asarray(stems, dtype=np.float64))
    out = np.empty((audio.shape[0], frame_count(audio.shape[1], n_fft, hop),
                    band_map.n_bands), dtype=np.float64)
    for index in range(audio.shape[0]):
        power = np.abs(stft(audio[index], n_fft, hop)) ** 2
        out[index] = band_map.band_db(power.T).T
    return out


def render_stems(stems, gains_db, band_map: BandMap, n_fft: int = DEFAULT_N_FFT,
                 hop: int = DEFAULT_HOP, spectra=None) -> np.ndarray:
    """Apply per-stem per-band gains and sum the stems back into a mono mix.

    ``gains_db`` is ``[n_stems, frames, bands]`` (or ``[n_stems, bands]`` for a
    static curve per stem).  ``spectra`` may pass pre-computed STFTs to avoid
    transforming the audio twice.
    """
    audio = np.atleast_2d(np.asarray(stems, dtype=np.float64))
    n_stems, n_samples = audio.shape
    gains = np.asarray(gains_db, dtype=np.float64)
    if gains.shape[0] != n_stems:
        raise ValueError(f"gains cover {gains.shape[0]} stems, audio has {n_stems}")
    mix = np.zeros(n_samples, dtype=np.float64)
    for index in range(n_stems):
        spec = spectra[index] if spectra is not None else stft(audio[index], n_fft, hop)
        mix += istft(apply_band_gains(spec, gains[index], band_map),
                     n_fft, hop, length=n_samples)
    return mix
