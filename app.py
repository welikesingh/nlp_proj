"""Fake News Detector - Streamlit app.

Uses a pre-trained TF-IDF vectorizer (vectorizer.pkl) and a
Multinomial Naive Bayes classifier (model.pkl). Place both .pkl files
in the same folder as this script, then run:

    streamlit run app.py
"""

from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict(text: str, vectorizer, model):
    features = vectorizer.transform([text])
    label = model.predict(features)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    known_words = features.nnz  # how many words the model recognised
    return label, probabilities, known_words


st.title("📰 Fake News Detector")
st.write("Paste a news headline or short article and the model will predict whether it looks **Real** or **Fake**.")

try:
    vectorizer, model = load_artifacts()
except FileNotFoundError:
    st.error("Could not find `vectorizer.pkl` and/or `model.pkl`. Put them in the same folder as `app.py`.")
    st.stop()

news_text = st.text_area(
    "News text",
    height=180,
    placeholder="e.g. Government launches new healthcare scheme for poor families",
)

if st.button("Check news", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, probs, known_words = predict(news_text, vectorizer, model)
        confidence = probs[label] * 100

        if label.lower() == "real":
            st.success(f"✅ This looks like **REAL** news ({confidence:.1f}% confidence)")
        else:
            st.error(f"🚨 This looks like **FAKE** news ({confidence:.1f}% confidence)")

        st.subheader("Class probabilities")
        for cls, p in probs.items():
            st.write(f"{cls}: {p * 100:.1f}%")
            st.progress(float(p))

        if known_words == 0:
            st.info(
                "None of the words in your text are in the model's vocabulary, "
                "so this prediction is based only on the prior class balance."
            )

with st.sidebar:
    st.header("About")
    st.write(
        "Model: TF-IDF vectorizer + Multinomial Naive Bayes (scikit-learn). "
        f"Vocabulary size: {len(vectorizer.vocabulary_)} words."
    )
    st.caption("The model was trained on a small dataset, so treat predictions as a demo, not a fact-check.")
