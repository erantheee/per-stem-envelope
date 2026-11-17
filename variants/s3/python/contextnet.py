"""ContextNet v0.2-s3 in pure numpy (no torch at runtime).

Layout: 92 -> 32 -> 16 -> 9, ReLU between layers, softmax on the last one.
3,657 parameters, 3,600 MACs per inference — the model fits an ESP32-S3 with
room to spare; the C port of this is a 40-line double loop.

Weights live in ``weights/contextnet_v0.2-s3.json`` (exported from the torch
checkpoint by ``tools/make_goldens.py``). Feature construction is *not* in this
module: the 92-dim window vector is built by the capture side (24 band means,
band std/slope, broadband, voice likelihood ...), see the exported
``feature_names`` in the JSON if you need the exact order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "weights/contextnet_v0.2-s3.json"


class ContextNet:
    def __init__(self, payload: dict) -> None:
        self.version = str(payload.get("version", "?"))
        self.classes = [str(c) for c in payload["classes"]]
        normalisation = payload["normalisation"]
        self.mu = np.asarray(normalisation["mu"], dtype=np.float64)
        self.sigma = np.asarray(normalisation["sigma"], dtype=np.float64)
        self.layers = [(np.asarray(layer["weight"], dtype=np.float64),
                        np.asarray(layer["bias"], dtype=np.float64))
                       for layer in payload["layers"]]
        self.input_dim = int(payload["input_dim"])
        if self.mu.shape != (self.input_dim,) or self.sigma.shape != (self.input_dim,):
            raise ValueError("normalisation vectors do not match input_dim")
        if self.layers[0][0].shape[1] != self.input_dim:
            raise ValueError("first layer does not match input_dim")
        if self.layers[-1][0].shape[0] != len(self.classes):
            raise ValueError("last layer does not match the class list")

    @classmethod
    def load(cls, path=None) -> "ContextNet":
        return cls(json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")))

    def _normalise(self, raw) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float64).reshape(-1)
        if x.size != self.input_dim:
            raise ValueError(f"expected {self.input_dim} features, got {x.size}")
        return (x - self.mu) / np.where(np.abs(self.sigma) < 1e-6, 1.0, self.sigma)

    def logits(self, raw_features) -> np.ndarray:
        x = self._normalise(raw_features)
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if index < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def probabilities(self, raw_features) -> np.ndarray:
        logits = self.logits(raw_features)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / exp.sum()

    def classify(self, raw_features):
        """Return ``(label, probability)`` for the winning class."""
        probs = self.probabilities(raw_features)
        index = int(np.argmax(probs))
        return self.classes[index], float(probs[index])
