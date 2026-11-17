"""Evidence: everything the decision layer knows about the *outside* world.

The point of this type is the boundary.  The algorithm does not care whether the
numbers come from a MEMS microphone, a phone, a file or a test, and it does not
know anything about the transport (JSON over HTTP, shared memory, message queue).
Adapters translate; this module only validates.

Fail-open is part of the contract: missing or malformed evidence must degrade to
"no change", never to a latched attenuation.
"""
from __future__ import annotations

import math

#: Levels below this are treated as "no spectrum" (an all-zero DMIC frame).
SILENT_SPECTRUM_DBFS = -119.0


class Evidence:
    """One snapshot of ambient conditions.

    Parameters
    ----------
    level_dbfs:
        Broadband ambient level (slow-weighted).  ``None`` means "no evidence".
    band_dbfs:
        Ambient spectrum, one value per control band (same band layout the
        :class:`~stemenvelope.bands.BandMap` uses).  Used for band dominance.
    event:
        Label for the dominant ambient event.  Free-form; the library only acts
        on the ones it knows (``keyboard``, ``speech``) and ignores the rest.
    voice:
        Voice likelihood 0..1 of the *ambient* signal, if a detector provides it.
    conversation:
        Conversation state of the surrounding product (``IDLE``, ``ANTICIPATING``,
        ``ENGAGED``, ``RELEASING``), or ``None``.
    """

    __slots__ = ("level_dbfs", "band_dbfs", "event", "voice", "conversation")

    def __init__(self, level_dbfs=None, band_dbfs=None, event=None,
                 voice: float = 0.0, conversation=None) -> None:
        self.level_dbfs = level_dbfs
        self.band_dbfs = band_dbfs
        self.event = event
        self.voice = voice
        self.conversation = conversation

    # ── validation ───────────────────────────────────────────────────────────
    def valid_level(self):
        """Broadband level as float, or ``None`` when unusable."""
        try:
            value = float(self.level_dbfs)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def valid_bands(self, n_bands: int):
        """Ambient spectrum as a tuple of ``n_bands`` floats, or ``None``."""
        if self.band_dbfs is None:
            return None
        try:
            values = tuple(float(v) for v in self.band_dbfs)
        except (TypeError, ValueError):
            return None
        if len(values) != int(n_bands):
            return None
        if not all(math.isfinite(v) for v in values):
            return None
        if max(values) <= SILENT_SPECTRUM_DBFS:
            return None                      # all-silent sentinel from the capture path
        return values

    def valid_voice(self) -> float:
        try:
            return max(0.0, min(1.0, float(self.voice)))
        except (TypeError, ValueError):
            return 0.0

    def valid_conversation(self) -> str:
        return str(self.conversation or "NONE").upper()

    def __repr__(self) -> str:
        return (f"Evidence(level_dbfs={self.level_dbfs!r}, event={self.event!r}, "
                f"voice={self.voice!r}, conversation={self.conversation!r})")
