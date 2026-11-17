"""The 24-band tuning recipe: ambient level + spectrum -> one gain curve.

This is the "what should happen to the music" layer.  It produces a single curve
in dB per band, and the per-stem envelope (see :mod:`stemenvelope.envelope`)
decides *which stem* actually receives it, in which band, and how fast.

The shape is a content prior damped by noise dominance, plus masking for known
interferers:

    g[b] = C[b] * exposure * (1 - k[b] * dominance[b]) + masking[b]

* ``C[b]``     — the intended tonal balance (a product decision, not a truth).
* ``exposure`` — how loud the room is, 0..1.
* ``k[b]``     — how much a noisy band should suppress the boost in that band.
* ``masking``  — extra lift around a keyboard or a background talker.

Every number below is a *calibration*, not a constant of nature.  They were
measured on one DMIC array on one board; a different microphone, gain staging or
mounting invalidates the dBFS anchors (``*_db_fs`` fields) even if the shape
stays usable.  Recalibrate the anchors before trusting the absolute numbers.
"""
from __future__ import annotations

import numpy as np

#: Intended tonal balance, 24 bands, dB.  Bass/lib presence forward, low-mid dip.
DEFAULT_CONTENT_CURVE_DB = np.array([
    2.5, 3.0, 0.0, -1.8, -1.8, -1.4, -0.6, -0.5,
    0.0, 0.5, 1.0, 1.2, 1.7, 2.1, 2.3, 2.3,
    1.9, 1.5, 1.3, 1.3, 1.3, 2.6, 2.6, 1.9,
])

#: Noise-dominance suppression per band (positive gains only are damped).
DEFAULT_MASK_K = np.array([
    0.65, 0.65, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.60, 0.60, 0.60, 0.45, 0.45, 0.45, 0.45, 0.45,
    0.45, 0.45, 0.60, 0.60, 0.60, 0.85, 0.85, 0.85,
])

#: Band index ranges for the masking emphasis regions (inclusive).
KEYBOARD_BANDS = (8, 16)     # ~1.0 - 3.4 kHz
SPEECH_BANDS = (8, 17)       # ~1.0 - 4.0 kHz
NOISE_BANDS = (11, 17)       # ~1.6 - 4.0 kHz


class TuningRecipe:
    """Ambient evidence -> a bounded per-band gain curve.

    Parameters are grouped by what they cost you if you change them:

    * shape (`content_curve_db`, `mask_k`, the four masking amounts) is portable
      across microphones and is the actual "tuning taste";
    * anchors (`noise_floor_dbfs`, `noise_ceil_dbfs`, `band_floor_dbfs`,
      `band_ceil_dbfs`) are hardware-specific and must be re-measured.
    """

    def __init__(self, content_curve_db=None, mask_k=None,
                 limit_db: float = 4.0,
                 noise_floor_dbfs: float = -66.0, noise_ceil_dbfs: float = -46.0,
                 band_floor_dbfs: float = -90.0, band_ceil_dbfs: float = -50.0,
                 keyboard_masking_db: float = 0.5, keyboard_emphasis_db: float = 1.5,
                 speech_masking_db: float = 1.0, speech_emphasis_db: float = 4.0,
                 noise_masking_db: float = 0.3, noise_emphasis_db: float = 1.0) -> None:
        self.content_curve_db = np.asarray(
            DEFAULT_CONTENT_CURVE_DB if content_curve_db is None else content_curve_db,
            dtype=np.float64)
        self.mask_k = np.asarray(
            DEFAULT_MASK_K if mask_k is None else mask_k, dtype=np.float64)
        if self.content_curve_db.shape != self.mask_k.shape:
            raise ValueError("content_curve_db and mask_k must have the same shape")
        self.n_bands = int(self.content_curve_db.size)
        self.limit_db = float(limit_db)
        self.noise_floor_dbfs = float(noise_floor_dbfs)
        self.noise_ceil_dbfs = float(noise_ceil_dbfs)
        self.band_floor_dbfs = float(band_floor_dbfs)
        self.band_ceil_dbfs = float(band_ceil_dbfs)
        self.keyboard_masking_db = float(keyboard_masking_db)
        self.keyboard_emphasis_db = float(keyboard_emphasis_db)
        self.speech_masking_db = float(speech_masking_db)
        self.speech_emphasis_db = float(speech_emphasis_db)
        self.noise_masking_db = float(noise_masking_db)
        self.noise_emphasis_db = float(noise_emphasis_db)

    # ── pieces ───────────────────────────────────────────────────────────────
    def exposure(self, level_dbfs) -> float:
        """0..1 loudness of the room against the calibrated span."""
        try:
            level = float(level_dbfs)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(level):
            return 1.0
        span = self.noise_ceil_dbfs - self.noise_floor_dbfs
        return float(min(1.0, max(0.0, (level - self.noise_floor_dbfs) / span)))

    def masking_db(self, level_dbfs, event=None) -> np.ndarray:
        """Extra lift for a known interferer (broadband + emphasis region)."""
        out = np.zeros(self.n_bands, dtype=np.float64)
        try:
            level = float(level_dbfs)
        except (TypeError, ValueError):
            return out
        depth = self.exposure(level)
        if depth <= 0.0:
            return out
        event = (event or "").lower()
        if event == "keyboard":
            out[:] = self.keyboard_masking_db * depth
            lo, hi = KEYBOARD_BANDS
            out[lo:hi + 1] += self.keyboard_emphasis_db * depth
        elif event == "speech":
            out[:] = self.speech_masking_db * depth
            lo, hi = SPEECH_BANDS
            out[lo:hi + 1] += self.speech_emphasis_db * depth
        else:
            out[:] = self.noise_masking_db * depth
            lo, hi = NOISE_BANDS
            out[lo:hi + 1] += self.noise_emphasis_db * depth
        return out

    def band_curve(self, evidence, with_masking: bool = True):
        """Evidence -> per-band gain (dB), or ``None`` when evidence is unusable.

        ``None`` is the fail-open signal: the caller must leave the audio alone.
        """
        if evidence is None:
            return None
        level = evidence.valid_level()
        if level is None:
            return None
        exposure = self.exposure(level)
        ambient = evidence.valid_bands(self.n_bands)
        if ambient is None:
            dominance = np.full(self.n_bands, exposure, dtype=np.float64)
        else:
            span = self.band_ceil_dbfs - self.band_floor_dbfs
            dominance = np.clip((np.asarray(ambient, dtype=np.float64)
                                 - self.band_floor_dbfs) / span, 0.0, 1.0)
        curve = self.content_curve_db * exposure * (1.0 - self.mask_k * dominance)
        curve = np.clip(curve, -self.limit_db, self.limit_db)
        if with_masking:
            curve = curve + self.masking_db(level, evidence.event)
        return np.clip(curve, -self.limit_db, self.limit_db)
