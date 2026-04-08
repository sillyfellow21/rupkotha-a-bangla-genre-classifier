from pathlib import Path

import streamlit as st

from src.inference import GenrePredictor

st.set_page_config(page_title="Rupkotha Genre Engine", page_icon="📚", layout="wide")

CUSTOM_CSS = """
<style>
.main {
    background: radial-gradient(circle at 0% 0%, #f3f7ff 0%, #fffef8 55%, #f8fff8 100%);
}
.hero {
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #083d77 0%, #0f6a8f 45%, #2f9e44 100%);
    color: white;
    margin-bottom: 1rem;
}
.pred-box {
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}
.pred-title {
    margin: 0 0 0.35rem 0;
    color: #0b2540;
    font-weight: 700;
}
.pred-meta {
    margin: 0;
    color: #334e68;
}
.pred-meta b {
    color: #0b2540;
}
.small-note {
    color: #5c6b73;
    font-size: 0.92rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
  <h1 style="margin-bottom:0.35rem;">Rupkotha Genre Engine</h1>
  <p style="margin:0;">Transformer-powered Bengali book genre prediction from summary text.</p>
</div>
""",
    unsafe_allow_html=True,
)

MODEL_DIR = Path("model/best_model")
LABEL_MAP = Path("artifacts/label_map.json")
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def has_model_artifacts() -> bool:
    """Check if all required model files exist."""
    # Verify both directory and critical model files
    return MODEL_DIR.exists() and all((MODEL_DIR / name).exists() for name in MODEL_FILES)


@st.cache_resource
def load_predictor() -> GenrePredictor:
    return GenrePredictor(
        model_dir=MODEL_DIR,
        label_map_path=LABEL_MAP,
        allow_fallback=True,
    )


examples = {
    "উদাহরণ ১ - ইতিবাচক": "এই বইটি সত্যিই অসাধারণ এবং অনুপ্রেরণাদায়ক। লেখক খুব সুন্দরভাবে গল্পটি উপস্থাপন করেছেন এবং আমি এটি সবাইকে সুপারিশ করব।",
    "উদাহরণ २ - নেতিবাচক": "বই পড়তে গিয়ে খুব হতাশ হয়েছি। গল্পের গতিপ্রবাহ খুবই ধীর এবং অনেক অংশ বোধগম্য নয়। এটি সময়ের অপচয় ছিল।",
    "উদাহরণ ३ - নিরপেক্ষ": "এই উপন্যাসটি মোটামুটি ভালো। কিছু অংশ আকর্ষণীয় ছিল কিন্তু অন্যান্য অংশ একটু দুর্বল মনে হয়েছে।",
}

left, right = st.columns([1.6, 1], gap="large")

with left:
    st.subheader("বইয়ের সারাংশ লিখুন")
    chosen_example = st.selectbox("নমুনা ইনপুট", ["নিজে লিখব"] + list(examples.keys()))

    default_text = ""
    if chosen_example != "নিজে লিখব":
        default_text = examples[chosen_example]

    user_text = st.text_area(
        "সারাংশ",
        value=default_text,
        height=220,
        placeholder="এখানে বাংলা বইয়ের সারাংশ লিখুন...",
    )

    predict_clicked = st.button("Predict Genre", type="primary", use_container_width=True)

with right:
    st.subheader("প্রকল্প তথ্য")
    st.markdown(
        """
- Model: BanglaBERT sequence classifier
- Task: Sentiment classification
- Classes: 3 labels (positive/neutral/negative)
- Input: Bengali Unicode text
- Output: Predicted sentiment + confidence
"""
    )
    st.markdown(
        '<p class="small-note">লোকাল মডেল না থাকলে fallback cloud model ব্যবহার হবে।</p>',
        unsafe_allow_html=True,
    )

    if not has_model_artifacts():
        st.warning("⚠️ লোকাল মডেল ফাইল পাওয়া যায়নি।")
        st.info(
            "Fallback cloud model ব্যবহার করা হবে। প্রথম prediction একটু সময় নিতে পারে।"
        )

if predict_clicked:
    if not user_text.strip():
        st.warning("অনুগ্রহ করে একটি সারাংশ লিখুন।")
    else:
        try:
            predictor = load_predictor()
            result = predictor.predict(user_text)
        except Exception as exc:
            st.error(f"Prediction ব্যর্থ হয়েছে: {exc}")
            st.info(
                "Network block থাকলে লোকাল trained model (`model/best_model`) যোগ করে আবার চেষ্টা করুন।"
            )
            st.stop()

        st.markdown("### Prediction")
        st.markdown(
            f"""
<div class="pred-box">
    <h3 class="pred-title">{result['predicted_genre']}</h3>
    <p class="pred-meta">Confidence: <b>{result['confidence'] * 100:.2f}%</b></p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(min(max(result["confidence"], 0.0), 1.0))

        st.markdown("### Class Probabilities")
        sorted_probs = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        for label, prob in sorted_probs:
            st.write(f"{label}: {prob * 100:.2f}%")
