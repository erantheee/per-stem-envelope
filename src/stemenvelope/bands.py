"""Log-spaced frequency band map (FFT bins <-> control bands).

The control surface is coarse on purpose: 24 third-octave-ish bands from 70 Hz to
13.6 kHz.  Perceptual gain smoothing does not need more resolution, and a coarse
map keeps the decision layer readable and cheap on small CPUs.

Band count is not fixed; pass your own ``centers_hz`` if you want 10, 24 or 31.
"""
from __future__ import annotations

import numpy as np

#: Default 24 centres, ~1/3 octave, 70.7 Hz .. 13.6 kHz at 44.1 kHz.
DEFAULT_CENTERS_HZ = np.array([
    70.7, 141.4, 244.9, 346.4, 451.7, 566.8, 696.5, 841.7,
    996.8, 1171.3, 1370.0, 1595.0, 1854.7, 2154.1, 2503.3,
    2916.3, 3414.7, 4036.1, 4829.1, 5824.1, 7020.0, 8552.8,
    10677.1, 13638.2,
])


class BandMap:
    """Maps FFT bins to control bands and back.

    ``band_power`` uses a nearest-centre membership matrix (each bin belongs to
    exactly one band), which is what a Goertzel-style analyser produces.  Gain
    *synthesis* interpolates linearly in log-frequency between centres so a band
    gain change does not leave a visible step at band borders.
    """

    def __init__(self, centers_hz=None, n_fft: int = 2048,
                 sample_rate: float = 44100.0) -> None:
        centers = DEFAULT_CENTERS_HZ if centers_hz is None else np.asarray(centers_hz, dtype=np.float64)
        if centers.ndim != 1 or centers.size < 2:
            raise ValueError("centers_hz must be a 1-D array with at least two centres")
        if not np.all(np.diff(centers) > 0):
            raise ValueError("centers_hz must be strictly increasing")
        if int(n_fft) < 8:
            raise ValueError("n_fft must be >= 8")
        self.centers_hz = centers
        self.n_bands = int(centers.size)
        self.n_fft = int(n_fft)
        self.sample_rate = float(sample_rate)

        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        log_freqs = np.log2(np.maximum(freqs, 1e-3))
        log_centers = np.log2(self.centers_hz)

        pos = np.interp(log_freqs, log_centers, np.arange(self.n_bands),
                        left=0.0, right=float(self.n_bands - 1))
        self.low = np.floor(pos).astype(np.int32)
        self.high = np.minimum(self.low + 1, self.n_bands - 1)
        self.frac = (pos - self.low).astype(np.float32)

        nearest = np.abs(log_freqs[:, None] - log_centers[None, :]).argmin(axis=1)
        self._membership = np.zeros((self.n_bands, freqs.size), dtype=np.float64)
        self._membership[nearest, np.arange(freqs.size)] = 1.0

    @property
    def n_bins(self) -> int:
        return self._membership.shape[1]

    def gains_to_bins(self, gains_db) -> np.ndarray:
        """Band gains (last axis = bands) -> per-bin linear-domain dB values."""
        gains = np.asarray(gains_db, dtype=np.float64)
        if gains.shape[-1] != self.n_bands:
            raise ValueError(f"expected {self.n_bands} band gains, got shape {gains.shape}")
        return gains[..., self.low] * (1.0 - self.frac) + gains[..., self.high] * self.frac

    def band_power(self, power) -> np.ndarray:
        """Bin power -> band power.  ``power`` is [bins] or [bins, frames]."""
        values = np.asarray(power, dtype=np.float64)
        if values.shape[0] != self.n_bins:
            raise ValueError(f"expected {self.n_bins} bins on axis 0, got {values.shape}")
        return self._membership @ values

    def band_db(self, power, floor: float = 1e-18) -> np.ndarray:
        """Bin power -> band dB (power is [bins] or [bins, frames])."""
        return 10.0 * np.log10(np.maximum(self.band_power(power), floor))

    def band_index(self, hz: float) -> int:
        """Index of the band closest to ``hz`` (log distance)."""
        return int(np.abs(np.log2(self.centers_hz) - np.log2(max(float(hz), 1e-3))).argmin())
