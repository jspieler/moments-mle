"""Photo quality gate.

Blocks uploads that are too blurry or too dark to be worth storing.

Based on handover/quality_gate_v2_FINAL.ipynb. Feature code copied out of the notebook
so the app does not need the notebook.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[2] / "handover" / "quality_gate.pkl"

_model = None


def _load():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as fh:
            _model = pickle.load(fh)
    return _model


def extract_features(image_bytes: bytes) -> list[float]:
    """Feature extraction, as used at upload time."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((512, 512)) # keep it fast for big uploads
    g = np.asarray(img.convert("L")).astype(float)
    lap = np.abs(np.diff(g, axis=0)[:, :-1] + np.diff(g, axis=1)[:-1, :])
    return [
        float(g.mean()),
        float(g.std()),
        float(lap.var()),
        float((g < 40).mean()),
    ]


def assess(image_bytes: bytes) -> tuple[float, bool]:
    """Return (usable_score, blocked)."""
    x = np.array([extract_features(image_bytes)])
    score = float(_load().predict_proba(x)[0, 1])
    return score, score < 0.5
