"""
Nexus-Kinematics Medical AI
============================

A chat-style interface around the EXISTING trained symptom-analysis
model (Random Forest, model/medical_symptom_model.pkl). This file does
not train, retrain, or modify the model in any way -- it only loads the
existing .pkl artifacts and builds a UI around them.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import base64

import streamlit as st

from backend import (
    ArtifactLoadError,
    LOGO_PATH,
    build_input_vector,
    load_disease_classes,
    load_metadata,
    load_model,
    load_symptom_names,
    predict_top_conditions,
)
from symptom_matcher import detect_symptoms

DISCLAIMER = (
    "This tool is an AI research/educational prototype and does not provide a "
    "medical diagnosis. AI-generated results may be inaccurate or incomplete. "
    "Please consult a qualified healthcare professional for medical advice. "
    "If this is a medical emergency, contact emergency services immediately."
)

APP_NAME = "Nexus-Kinematics"
APP_TITLE = "Nexus-Kinematics Medical AI"


# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩺",
        layout="centered",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; max-width: 780px; }
            .nk-subtitle {
                text-align: center;
                color: var(--text-color, #6b7280);
                font-size: 0.95rem;
                margin-top: -0.6rem;
                margin-bottom: 1.4rem;
            }
            .nk-app-name {
                text-align: center;
                font-size: 1.6rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                margin-bottom: 0;
            }
            .nk-app-tagline {
                text-align: center;
                font-size: 1.05rem;
                font-weight: 500;
                opacity: 0.85;
                margin-top: -0.2rem;
            }
            .nk-disclaimer {
                background: rgba(255, 176, 32, 0.12);
                border: 1px solid rgba(255, 176, 32, 0.35);
                border-radius: 10px;
                padding: 0.75rem 1rem;
                font-size: 0.85rem;
                margin-bottom: 1.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    if LOGO_PATH.exists():
        logo_bytes = LOGO_PATH.read_bytes()
        encoded = base64.b64encode(logo_bytes).decode()
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; margin-bottom: 0.6rem;">
                <img src="data:image/jpeg;base64,{encoded}"
                     style="width:120px; height:120px; object-fit:contain;
                            border-radius:50%;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Logo file not found -- continuing without it.", icon="⚠️")

    st.markdown(f'<div class="nk-app-name">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="nk-app-tagline">Medical AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nk-subtitle">Describe your symptoms and get model generated possible conditions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="nk-disclaimer">⚕️ {DISCLAIMER}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Prediction / response formatting
# ---------------------------------------------------------------------------

def format_assistant_reply(detected: list[str], results: list[dict]) -> str:
    lines: list[str] = []

    if not detected:
        lines.append(
            "I couldn't confidently match any known symptoms in that message. "
            "Could you describe what you're feeling a bit more specifically "
            "(e.g. \"fever, headache, and nausea\")?"
        )
        return "\n".join(lines)

    lines.append("**Detected symptoms:**")
    for s in detected:
        lines.append(f"- {s.capitalize()}")

    lines.append("")
    lines.append("**AI-generated possible conditions:**")
    for i, result in enumerate(results, start=1):
        condition = str(result["condition"]).capitalize()
        confidence = result["confidence"]
        if confidence is None:
            lines.append(f"{i}. {condition}")
        else:
            lines.append(f"{i}. {condition} -- {confidence:.1f}% confidence")

    lines.append("")
    lines.append(f"*{DISCLAIMER}*")
    return "\n".join(lines)


def run_prediction_pipeline(
    user_message: str,
    model,
    symptom_names: list[str],
    disease_classes: list[str],
) -> str:
    detected = detect_symptoms(user_message, symptom_names)
    if not detected:
        return format_assistant_reply(detected, [])

    try:
        input_vector = build_input_vector(symptom_names, detected)
        results = predict_top_conditions(model, input_vector, disease_classes, top_n=3)
    except Exception as exc:  # noqa: BLE001
        return (
            "Sorry -- I detected some symptoms, but ran into a problem generating "
            f"a prediction from the model ({exc.__class__.__name__}). "
            "Please try rephrasing your message."
        )

    return format_assistant_reply(detected, results)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    configure_page()
    render_header()

    # Load existing artifacts (cached -- loaded once per session, not per message).
    try:
        model = load_model()
        symptom_names = load_symptom_names()
        disease_classes = load_disease_classes()
        metadata = load_metadata()
    except ArtifactLoadError as exc:
        st.error(f"Couldn't start the assistant: {exc}")
        st.stop()
        return
    with st.sidebar:
        st.subheader(APP_NAME)
        st.caption("Medical AI Assistant")

        if metadata:
            st.markdown("**Model info**")
            st.markdown(
                f"- Type: {metadata.get('model_type', 'Unknown')}\n"
                f"- Symptoms known: {metadata.get('number_of_symptoms', len(symptom_names))}\n"
                f"- Conditions covered: {metadata.get('number_of_diseases', len(disease_classes))}\n"
                f"- Test accuracy: {metadata.get('test_accuracy', 0) * 100:.1f}%\n"
                f"- Weighted F1: {metadata.get('weighted_f1', 0):.2f}"
            )
            st.caption(
                "This is a research/educational prototype, not a clinically "
                "validated diagnostic system."
            )

        with st.expander("Supported Symptoms"):
            st.caption(
                f"This model supports {len(symptom_names)} trained symptoms."
            )

            query = st.text_input(
                "Search symptoms",
                label_visibility="collapsed",
                placeholder="Search e.g. headache, chest, nausea..."
            )

            if query:
                matches = [
                    s for s in symptom_names
                    if query.lower() in s.lower()
                ]

                if matches:
                    st.markdown(
                        "\n".join(
                            f"- {s.replace('_', ' ').capitalize()}"
                            for s in matches
                        )
                    )
                else:
                    st.info("No supported symptom matches found.")
            else:
                st.markdown(
                    "\n".join(
                        f"- {s.replace('_', ' ').capitalize()}"
                        for s in symptom_names
                    )
                )

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Describe your symptoms...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing symptoms..."):
                reply = run_prediction_pipeline(
                    user_message, model, symptom_names, disease_classes
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
