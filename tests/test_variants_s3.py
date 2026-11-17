import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import contextnet as cn
import env_map

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/contextnet_golden.json"


@pytest.fixture(scope="module")
def net():
    return cn.ContextNet.load()


def test_model_shape_and_metadata(net):
    assert net.input_dim == 92
    assert len(net.classes) == 9
    assert net.layers[0][0].shape == (32, 92)
    assert net.layers[1][0].shape == (16, 32)
    assert net.layers[2][0].shape == (9, 16)


def test_logits_match_the_original_torch_model(net):
    pairs = json.loads(GOLDEN.read_text(encoding="utf-8"))["pairs"]
    assert len(pairs) >= 3
    for pair in pairs:
        mine = net.logits(pair["raw_features"])
        theirs = np.asarray(pair["logits"])
        assert np.allclose(mine, theirs, atol=2e-4), f"max err {np.abs(mine - theirs).max()}"


def test_probabilities_are_a_distribution(net):
    pairs = json.loads(GOLDEN.read_text(encoding="utf-8"))["pairs"]
    probs = net.probabilities(pairs[0]["raw_features"])
    assert probs.shape == (9,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)


def test_classify_returns_a_known_label(net):
    pairs = json.loads(GOLDEN.read_text(encoding="utf-8"))["pairs"]
    label, probability = net.classify(pairs[0]["raw_features"])
    assert label in net.classes
    assert 0.0 < probability <= 1.0


def test_rejects_wrong_feature_count(net):
    with pytest.raises(ValueError):
        net.logits(np.zeros(91))


def test_env_map_covers_every_class(net):
    for label in net.classes:
        assert env_map.env_for_class(label) in (
            "quiet", "speech", "babble", "street", "office", "wind", "keyboard")


def test_env_map_documented_choices():
    assert env_map.env_for_class("music") == "quiet"      # content, not a hazard
    assert env_map.env_for_class("near_voice") == "speech"
    assert env_map.env_for_class("horn") == "office"
    assert env_map.env_for_class("nonsense") == "quiet"   # fail-open


def test_weak_winner_fails_open(net):
    probs = [0.02] * 9
    env, label, probability = env_map.env_from_probs(net.classes, probs)
    assert (env, label) == ("quiet", "uncertain")
    env2, label2, _ = env_map.env_from_probs(net.classes, [0.9] + [0.0125] * 8)
    assert label2 == net.classes[0] and env2 == env_map.env_for_class(net.classes[0])


def test_env_from_probs_rejects_bad_shapes(net):
    assert env_map.env_from_probs(net.classes, []) == ("quiet", "invalid", 0.0)
    assert env_map.env_from_probs(net.classes, [0.5]) == ("quiet", "invalid", 0.0)


@pytest.mark.skipif(not shutil.which("cc") or not shutil.which("make"),
                    reason="needs a C compiler and make")
def test_c_port_matches_python_reference():
    result = subprocess.run(["make", "-C", str(ROOT / "variants/s3"), "check"],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "parity" in result.stdout
