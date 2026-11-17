"""Per-stem, per-band gain envelope with content gating.

This is the part that makes the result "per stem" instead of "a curve pasted on
the mix".  Two ideas, both small:

**Content gating.**  A stem only receives the curve in the bands where that stem
has content.  ``presence = clip((band_db - (peak_db - window_db)) / window_db)``
is 1.0 at the stem's loudest band and fades to 0.0 ``window_db`` below it.  Gain
is then ``curve * presence``.  Without this, a 3 dB lift intended for a vocal
lands on every stem's empty band, and separation residue (what the separator did
not subtract cleanly) gets lifted into an audible artefact.

**Independent smoothing per stem and band.**  One first-order lag per (stem,
band) cell.  The curve can jump when the noise label changes; the applied gain
cannot, so it does not pump.
"""
from __future__ import annotations

import numpy as np

from .smoothing import slew


class PerStemEnvelope:
    """Holds one gain per (stem, band) and moves it toward a gated target.

    Parameters
    ----------
    n_stems, n_bands:
        Grid size.  Stems are whatever the separator emits (4, 6, ...).
    tau_rise_s, tau_fall_s:
        Time constants.  ``tau_rise_s`` is the path toward a larger (less cutting)
        gain, ``tau_fall_s`` toward a smaller one.  Defaults (0.65 / 0.85 s)
        mirror a released product; shorter values feel snappier but let loudness
        follow the label, longer ones feel stable but late.
    presence_window_db:
        Dynamic range below a stem's peak that still counts as "this stem is
        here".  24 dB keeps quiet-but-present bands in play; 12 dB is nearly a
        hard mask; 40 dB is close to no gating at all.
    silence_floor_db:
        Absolute floor: a stem whose loudest band is below this is treated as
        absent and receives no gain.  Without it a digitally silent stem still has
        ``peak - window`` below its own peak, so relative gating alone would hand
        it the full curve.  ``-80`` dB of band power is far below anything
        audible and far above a numerical zero (``-180`` dB), so the gap is wide
        and the value is not delicate.
    max_boost_db, max_cut_db:
        Safety bounds on the applied gain, independent of the curve.
    """

    def __init__(self, n_stems: int, n_bands: int = 24,
                 tau_rise_s: float = 0.65, tau_fall_s: float = 0.85,
                 presence_window_db: float = 24.0, silence_floor_db: float = -80.0,
                 max_boost_db: float = 3.0, max_cut_db: float = -12.0) -> None:
        if int(n_stems) < 1 or int(n_bands) < 1:
            raise ValueError("n_stems and n_bands must be >= 1")
        if float(presence_window_db) <= 0.0:
            raise ValueError("presence_window_db must be > 0")
        if float(max_cut_db) > 0.0 or float(max_boost_db) < 0.0:
            raise ValueError("expected max_cut_db <= 0 <= max_boost_db")
        self.n_stems = int(n_stems)
        self.n_bands = int(n_bands)
        self.tau_rise_s = float(tau_rise_s)
        self.tau_fall_s = float(tau_fall_s)
        self.presence_window_db = float(presence_window_db)
        self.silence_floor_db = float(silence_floor_db)
        self.max_boost_db = float(max_boost_db)
        self.max_cut_db = float(max_cut_db)
        self.gains_db = np.zeros((self.n_stems, self.n_bands), dtype=np.float64)

    def reset(self, n_stems=None) -> None:
        """Zero the envelope, optionally retargeting a different stem layout."""
        if n_stems is not None:
            if int(n_stems) < 1:
                raise ValueError("n_stems must be >= 1")
            self.n_stems = int(n_stems)
        self.gains_db = np.zeros((self.n_stems, self.n_bands), dtype=np.float64)

    # ── gating ───────────────────────────────────────────────────────────────
    def presence(self, stem_band_db) -> np.ndarray:
        """Content gate per (stem, band), 0..1, from that stem's own band levels."""
        content = np.atleast_2d(np.asarray(stem_band_db, dtype=np.float64))
        if content.shape != (self.n_stems, self.n_bands):
            raise ValueError(
                f"stem_band_db must be [{self.n_stems}, {self.n_bands}], got {content.shape}")
        finite = np.where(np.isfinite(content), content, -240.0)
        peak = np.max(finite, axis=1, keepdims=True)
        floor = peak - self.presence_window_db
        gated = np.clip((finite - floor) / self.presence_window_db, 0.0, 1.0)
        # A stem whose loudest band is below the absolute floor is absent, not
        # "quiet everywhere".  Relative gating alone would give it the full curve.
        gated[np.asarray(peak[:, 0] < self.silence_floor_db)] = 0.0
        return gated

    def target(self, stem_band_db, curve_db) -> np.ndarray:
        """Gated, bounded target gains for one decision tick."""
        curve = np.asarray(curve_db, dtype=np.float64)
        if curve.shape != (self.n_bands,):
            raise ValueError(f"curve_db must be [{self.n_bands}], got {curve.shape}")
        bounded = np.clip(curve, self.max_cut_db, self.max_boost_db)
        return np.clip(self.presence(stem_band_db) * bounded[None, :],
                       self.max_cut_db, self.max_boost_db)

    # ── update ───────────────────────────────────────────────────────────────
    def step(self, stem_band_db, curve_db, dt_s: float, active: bool = True,
             tau_rise_s=None, tau_fall_s=None) -> np.ndarray:
        """Advance one tick and return the applied gains ``[n_stems, n_bands]``.

        ``active=False`` (stale or missing evidence) drives every gain toward 0 dB
        instead of freezing, so a lost evidence stream returns to transparency
        rather than to a latched cut.
        """
        if active:
            target = self.target(stem_band_db, curve_db)
        else:
            target = np.zeros_like(self.gains_db)
        self.gains_db = slew(
            self.gains_db, target, dt_s,
            self.tau_rise_s if tau_rise_s is None else tau_rise_s,
            self.tau_fall_s if tau_fall_s is None else tau_fall_s)
        return self.gains_db.copy()

    def peak_db(self) -> np.ndarray:
        """Strongest |gain| per stem (telemetry / assertions)."""
        return np.max(np.abs(self.gains_db), axis=1)
