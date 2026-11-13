from pathlib import Path
import logging

import streamlit as st

from engine import LANGUAGES, StoryEngine


st.set_page_config(
    page_title="Tiny African Stories",
    page_icon="📖",
    layout="centered",
)

MODEL_FOLDER = Path(__file__).resolve().parent / "model"

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8f5;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
    }

    h1 {
        letter-spacing: -0.045em;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #e1e7df;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    return StoryEngine(MODEL_FOLDER)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Story settings")

    language = st.selectbox(
        "Response language",
        LANGUAGES,
    )

    st.caption(
        "Auto leaves the language choice to the model "
        "and omits the target-language field."
    )

    max_tokens = st.slider(
        "Maximum new tokens",
        min_value=32,
        max_value=384,
        value=128,
        step=16,
    )

    with st.expander("Generation settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.2,
            value=0.7,
            step=0.05,
        )

        repetition_penalty = st.slider(
            "Repetition penalty",
            min_value=1.0,
            max_value=1.4,
            value=1.1,
            step=0.05,
        )

        st.caption(
            "For greedy evaluation, use temperature 0 "
            "and repetition penalty 1."
        )

    if st.button(
        "Clear stories",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("BULaMU-V3 · 42M parameters")
    st.caption("Seven languages · Experimental model")


st.title("Tiny African Stories")
st.caption("powered by BULaMU-V3")

st.caption(
    "Outputs may repeat, drift from your request, or contain "
    "language errors. Each request is independent."
)


engine = None

try:
    with st.spinner(
        "Loading BULaMU Story and its custom tokenizer…"
    ):
        engine = load_model()

    st.caption(
        f"Ready · {engine.device.upper()} · "
        f"{engine.context:,}-token context"
    )

except Exception as error:
    logging.exception("Model loading failed")

    st.error(
        "Could not load the model from the model/ folder."
    )

    with st.expander("Error details"):
        st.code(str(error))


if not st.session_state.messages:
    st.markdown(
        "**Try:** Write a short story about a young farmer "
        "who helps his neighbor."
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["text"])

        if message["role"] == "assistant":
            st.caption(message["caption"])

            with st.expander("Prompt sent to the model"):
                st.code(
                    message["prompt"],
                    language=None,
                )


# Native Streamlit chat input includes a send-arrow button.
instruction = st.chat_input(
    "What story would you like to hear?",
    disabled=engine is None,
    max_chars=6000,
)


if instruction:
    with st.chat_message("user"):
        st.write(instruction)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Writing your story…"):
                result = engine.generate(
                    instruction=instruction,
                    language=language,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

            answer = result["text"]

            if not answer:
                answer = (
                    "The model ended without producing a story. "
                    "Try another prompt or generation setting."
                )

            caption = (
                f"{language} · "
                f"{result['output_tokens']} tokens · "
                f"{result['seconds']:.1f}s"
            )

            st.write(answer)
            st.caption(caption)

            if result["budget_reduced"]:
                st.caption(
                    "Output length was reduced to fit "
                    "the model's context window."
                )

            with st.expander("Prompt sent to the model"):
                st.code(
                    result["prompt"],
                    language=None,
                )

        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "text": instruction,
                },
                {
                    "role": "assistant",
                    "text": answer,
                    "caption": caption,
                    "prompt": result["prompt"],
                },
            ]
        )

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        logging.exception("Story generation failed")

        st.error(
            "Story generation failed. "
            "Check the error details and deployment logs."
        )

        with st.expander("Error details"):
            st.code(str(error))