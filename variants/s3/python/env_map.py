"""Map ContextNet's 9 context classes onto the envelope's 7 environment labels.

The classifier and the policy answer different questions:

* the classifier says *what the room sounds like* (music, horn, street, ...);
* the policy says *what to do to the music* and only knows
  ``quiet/speech/babble/street/office/wind/keyboard``.

This module is the seam, and the mapping is a **policy choice**, not a fact:

* ``music`` -> ``quiet`` because music is content, not an ambient hazard. Mapping
  it to anything else would attenuate the very thing we are playing.
* ``horn``/``steady``/``other`` -> ``office``: broadband, steady, unremarkable
  noise. Keeping them together avoids three near-identical modifier rows.
* ``near_voice`` -> ``speech``: a talker is present, so the intent-driven branch
  applies (level-independent, per decision.apply_environment).
* ``quiet``/``babble``/``street``/``wind`` map to themselves.

Change the table, then re-run the demo — but change it deliberately.
"""
from __future__ import annotations

DEFAULT_CLASS_TO_ENV = {
    "quiet": "quiet",
    "music": "quiet",
    "babble": "babble",
    "street": "street",
    "wind": "wind",
    "near_voice": "speech",
    "horn": "office",
    "steady": "office",
    "other": "office",
}

#: Below this confidence the label is not trusted and the envelope stays flat.
DEFAULT_MIN_PROBABILITY = 0.45


def env_for_class(label: str, mapping=None) -> str:
    table = DEFAULT_CLASS_TO_ENV if mapping is None else mapping
    return table.get(str(label).lower(), "quiet")


def env_from_probs(classes, probabilities, mapping=None,
                   min_probability: float = DEFAULT_MIN_PROBABILITY):
    """Pick the winning class, then translate it to an environment label.

    Returns ``(env, label, probability)``.  A weak winner returns
    ``("quiet", "uncertain", p)`` — fail-open, never a latched attenuation.
    """
    probabilities = list(probabilities)
    if not probabilities or len(probabilities) != len(classes):
        return "quiet", "invalid", 0.0
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    probability = float(probabilities[index])
    if probability < float(min_probability):
        return "quiet", "uncertain", probability
    label = str(classes[index])
    return env_for_class(label, mapping), label, probability
