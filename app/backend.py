"""
backend.py

Loads the EXISTING trained artifacts (model, symptom vocabulary,
disease classes, metadata) and runs predictions against them.

This module never trains, retrains, or mutates the model in any way.
It only reads the .pkl files that already exist under /model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

# Project layout (robust to being moved / run from any working directory,
# and to being run on Windows or elsewhere).
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
ASSETS_DIR = APP_DIR / "assests"

MODEL_PATH = MODEL_DIR / "medical_symptom_model.pkl"
SYMPTOM_NAMES_PATH = MODEL_DIR / "symptom_names.pkl"
DISEASE_CLASSES_PATH = MODEL_DIR / "disease_classes.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.pkl"
LOGO_PATH = ASSETS_DIR / "logo.jpeg"


class ArtifactLoadError(Exception):
    """Raised when a required existing artifact can't be loaded."""


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load the existing trained model. Never retrains it."""
    if not MODEL_PATH.exists():
        raise ArtifactLoadError(
            f"Model file not found at '{MODEL_PATH.name}'. "
            "Expected it under the project's /model folder."
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(
            f"Could not load the trained model ({exc.__class__.__name__}: {exc}). "
            "This is often caused by a scikit-learn version mismatch between "
            "training and this environment."
        ) from exc


@st.cache_resource(show_spinner=False)
def load_symptom_names() -> list[str]:
    if not SYMPTOM_NAMES_PATH.exists():
        raise ArtifactLoadError(f"'{SYMPTOM_NAMES_PATH.name}' not found in /model.")
    try:
        return list(joblib.load(SYMPTOM_NAMES_PATH))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(f"Could not load symptom vocabulary: {exc}") from exc


@st.cache_resource(show_spinner=False)
def load_disease_classes() -> list[str]:
    if not DISEASE_CLASSES_PATH.exists():
        raise ArtifactLoadError(f"'{DISEASE_CLASSES_PATH.name}' not found in /model.")
    try:
        return list(joblib.load(DISEASE_CLASSES_PATH))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(f"Could not load disease classes: {exc}") from exc


@st.cache_resource(show_spinner=False)
def load_metadata() -> Optional[dict]:
    if not METADATA_PATH.exists():
        return None
    try:
        return dict(joblib.load(METADATA_PATH))
    except Exception:  # noqa: BLE001
        return None


def build_input_vector(symptom_names: list[str], detected_symptoms: list[str]) -> pd.DataFrame:
    """
    Build the exact one-hot input format the model was trained on:
    a single-row DataFrame, columns = symptom_names (in order),
    dtype uint8, 1 where a symptom was detected, else 0.
    """
    input_data = pd.DataFrame(0, index=[0], columns=symptom_names, dtype="uint8")
    for symptom in detected_symptoms:
        if symptom in input_data.columns:
            input_data.loc[0, symptom] = 1
    return input_data


def predict_top_conditions(
    model,
    input_vector: pd.DataFrame,
    disease_classes: list[str],
    top_n: int = 3,
) -> list[dict]:
    """
    Run the EXISTING model on `input_vector` and return the top_n
    predicted conditions with confidence, using the model's own
    predict_proba output. No hard-coded or invented results.
    """
    classes = list(getattr(model, "classes_", disease_classes))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_vector)[0]
        top_indices = probabilities.argsort()[-top_n:][::-1]
        return [
            {
                "condition": classes[i],
                "confidence": float(probabilities[i]) * 100,
            }
            for i in top_indices
        ]

    # Fallback for a model without predict_proba: single hard prediction,
    # no fabricated confidence value.
    prediction = model.predict(input_vector)[0]
    return [{"condition": prediction, "confidence": None}]
