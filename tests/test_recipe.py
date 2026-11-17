import numpy as np
import pytest

from stemenvelope.evidence import Evidence
from stemenvelope.recipe import TuningRecipe


def _recipe():
    return TuningRecipe()


def test_missing_level_fails_open():
    assert _recipe().band_curve(Evidence(level_dbfs=None)) is None
    assert _recipe().band_curve(None) is None


def test_quiet_room_is_flat():
    curve = _recipe().band_curve(Evidence(level_dbfs=-90.0))
    assert np.allclose(curve, 0.0)


def test_loud_room_is_bounded_and_non_zero():
    curve = _recipe().band_curve(Evidence(level_dbfs=-40.0))
    assert np.max(np.abs(curve)) > 0.0
    assert np.all(np.abs(curve) <= 4.0 + 1e-9)


def test_keyboard_lifts_its_emphasis_region():
    recipe = _recipe()
    plain = recipe.band_curve(Evidence(level_dbfs=-50.0))
    keyboard = recipe.band_curve(Evidence(level_dbfs=-50.0, event="keyboard"))
    assert np.all(keyboard[8:17] > plain[8:17])


def test_speech_lifts_more_than_noise():
    recipe = _recipe()
    noise = recipe.band_curve(Evidence(level_dbfs=-50.0, event=None))
    speech = recipe.band_curve(Evidence(level_dbfs=-50.0, event="speech"))
    assert np.max(speech) > np.max(noise)


def test_noisy_band_suppresses_its_own_boost():
    recipe = _recipe()
    n_bands = recipe.n_bands
    quiet_bands = [-90.0] * n_bands
    loud_band_index = 1
    loud_bands = list(quiet_bands)
    loud_bands[loud_band_index] = -40.0
    a = recipe.band_curve(Evidence(level_dbfs=-50.0, band_dbfs=quiet_bands))
    b = recipe.band_curve(Evidence(level_dbfs=-50.0, band_dbfs=loud_bands))
    assert b[loud_band_index] < a[loud_band_index]


def test_masking_can_be_disabled():
    recipe = _recipe()
    with_mask = recipe.band_curve(Evidence(level_dbfs=-50.0, event="speech"))
    without = recipe.band_curve(Evidence(level_dbfs=-50.0, event="speech"), with_masking=False)
    assert np.all(without <= with_mask + 1e-12)


def test_recipe_validation():
    with pytest.raises(ValueError):
        TuningRecipe(content_curve_db=np.zeros(24), mask_k=np.zeros(23))


def test_exposure_clamps():
    recipe = _recipe()
    assert recipe.exposure(-100.0) == 0.0
    assert recipe.exposure(-30.0) == 1.0
    assert recipe.exposure(None) == 1.0
