"""Ambient state -> per-stem attenuation policy, with hysteresis.

Two layers, kept deliberately separate:

1. **Classification** (where does the noise come from?) is *not* in this package.
   A listening model, a small classifier or a human label produces one of
   :data:`ENVIRONMENTS`.  This package assumes that label exists.

2. **Policy** (what does that label do to the music?) is here.  It is a table of
   attenuation-only offsets plus two scalers: the panel gear (how strongly to
   react) and the measured noise strength (0..1).

Attenuation-only is a design decision, not an oversight.  Boosting estimated
stems re-amplifies separation leakage; cutting a competitor is always safer than
lifting the target when both come from the same imperfect separator.
"""
from __future__ import annotations

#: Labels the policy understands.  Anything else fails open to ``quiet``.
ENVIRONMENTS = ("quiet", "speech", "babble", "street", "office", "wind", "keyboard")

#: Attenuation-only offsets (dB) applied on top of the content profile.
#:
#: Defaults are intentionally conservative: a labelled background state does not
#: touch the mix, because the honest answer on our hardware was "the envelope
#: layer already handles it, and cutting drums to protect vocals is a taste
#: decision, not an algorithm decision".  Masking-type events (keyboard, a nearby
#: talker) are the ones that get an actual offset.
DEFAULT_MODIFIERS = {
    "quiet": {},
    "speech": {"vocals": 1.0, "drums": 0.5},
    "babble": {},
    "street": {},
    "office": {},
    "wind": {},
    "keyboard": {"vocals": 1.5, "drums": 1.5, "bass": 0.6,
                 "guitar": 0.6, "piano": 0.6, "other": 0.3},
}

#: States whose strength follows the measured ambient level.  ``speech`` is
#: intent-driven (a talker is present or not) and stays level-independent.
NOISE_ENVS = ("babble", "street", "office", "wind", "keyboard")

#: Level mapping for the continuous noise response (dBFS, broadband).
NOISE_FLOOR_DBFS = -66.0
NOISE_CEIL_DBFS = -46.0

#: A labelled noise state always pulls at least this fraction of its table; the
#: rest scales with level.  Without it "labelled noise but the room reads quiet"
#: would collapse to a no-op.
NOISE_BASE_FACTOR = 0.7

#: Floor for a composed target so one bad classification cannot mute a stem.
ENV_TARGET_FLOOR_DB = -9.0

#: Ceiling for the composed target (offsets are attenuation-only, but the content
#: profile can carry a boost and must stay bounded).
ENV_TARGET_CEIL_DB = 1.5


def noise_exposure(level_dbfs) -> float:
    """0..1 ambient-noise strength from the broadband level.

    Missing or non-finite level fails **open** to 1.0: a schema change must never
    silently mute the environment layer.
    """
    if level_dbfs is None:
        return 1.0
    try:
        value = float(level_dbfs)
    except (TypeError, ValueError):
        return 1.0
    span = NOISE_CEIL_DBFS - NOISE_FLOOR_DBFS
    return float(min(1.0, max(0.0, (value - NOISE_FLOOR_DBFS) / span)))


class EnvironmentTracker:
    """Hysteresis over environment labels.

    A candidate must persist ``hold_windows`` consecutive windows before it
    replaces the current state, so a single misclassified window cannot pump the
    gains.  ``speech`` may enter faster (``fast_windows``) because missing an
    active talker costs more than releasing late; its release still needs the
    full hold.

    Missing or unknown evidence fails open to ``quiet`` (= identity), never to a
    latched attenuation.
    """

    def __init__(self, hold_windows: int = 3, fast_windows: int = 2) -> None:
        self.hold_windows = max(int(hold_windows), 1)
        self.fast_windows = max(int(fast_windows), 1)
        self.state = None
        self.source = "unknown"
        self._pending = None
        self._held = 0

    def update(self, env) -> str:
        self.source = env if env in ENVIRONMENTS else "unknown"
        if env not in ENVIRONMENTS:
            self.state, self._pending, self._held = "quiet", None, 0
            return self.state
        if self.state is None:
            self.state = env
            return self.state
        if env == self.state:
            self._pending, self._held = None, 0
            return self.state
        threshold = self.fast_windows if env == "speech" else self.hold_windows
        if env == self._pending:
            self._held += 1
        else:
            self._pending, self._held = env, 1
        if self._held >= threshold:
            self.state, self._pending, self._held = env, None, 0
        return self.state


def apply_environment(profile, env, env_scale: float = 1.0,
                      noise_scale: float = 1.0, modifiers=None) -> dict:
    """Compose a per-stem profile with the environment contrast.

    Parameters
    ----------
    profile:
        Base per-stem gains in dB (0.0 for "leave alone").  Pass ``{}`` to use the
        environment layer on its own.
    env:
        Environment label; unknown labels behave like ``quiet``.
    env_scale:
        Gear / aggressiveness multiplier on the dynamic response (1.0 = base).
    noise_scale:
        0..1 measured noise strength.  Noise-driven states keep
        :data:`NOISE_BASE_FACTOR` so a labelled state can never read as zero.
    modifiers:
        Optional override of :data:`DEFAULT_MODIFIERS`.

    Returns
    -------
    dict
        ``{stem: gain_db}`` on the keys seen in ``profile`` or the table.
    """
    table = DEFAULT_MODIFIERS if modifiers is None else modifiers
    state = env if env in ENVIRONMENTS else "quiet"
    modifier = table.get(state, {})
    if state in NOISE_ENVS:
        factor = NOISE_BASE_FACTOR + (1.0 - NOISE_BASE_FACTOR) * float(noise_scale)
        scale = float(env_scale) * factor
    elif state == "speech":
        scale = 1.0
    else:
        scale = float(env_scale)
    floor_db = ENV_TARGET_FLOOR_DB * float(env_scale)
    stems = set(profile) | set(modifier)
    return {
        stem: max(floor_db, min(ENV_TARGET_CEIL_DB,
                                float(profile.get(stem, 0.0))
                                + float(modifier.get(stem, 0.0)) * scale))
        for stem in sorted(stems)
    }
