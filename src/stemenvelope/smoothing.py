"""Attack/release smoothing shared by the decision and envelope layers."""
from __future__ import annotations

import numpy as np


def slew(current, target, dt_s, tau_rise_s, tau_fall_s):
    """One first-order exponential step toward ``target`` (scalar or array).

    ``tau_rise_s`` applies while the value moves toward a *larger* number (a less
    attenuated gain), ``tau_fall_s`` while it moves toward a *smaller* one (more
    attenuation).  One tau reaches ~63 % of a step.

    This is deliberately the whole smoothing model: 24 independent band gains with
    a scalar time constant per direction.  No attack/hold/release state machine,
    no lookahead — both are worth adding only once a specific artefact demands it.
    """
    cur = np.asarray(current, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    dt = max(float(dt_s), 0.0)
    tau = np.where(tgt > cur, max(float(tau_rise_s), 1e-6), max(float(tau_fall_s), 1e-6))
    alpha = 1.0 - np.exp(-dt / tau)
    out = cur + (tgt - cur) * alpha
    return float(out) if out.ndim == 0 else out


def time_constant_reached(tau_s: float, fraction: float = 0.63) -> float:
    """How long a first-order lag needs to cover ``fraction`` of a step."""
    return -float(tau_s) * np.log(max(1e-9, 1.0 - float(fraction)))
