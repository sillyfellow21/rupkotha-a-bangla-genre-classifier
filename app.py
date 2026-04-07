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
    border: 1px solid #d9e2ec;
    border-radius: 14px;
    padding: 1rem;
    background: #ffffff;
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


@st.cache_resource
def load_predictor() -> GenrePredictor:
    return GenrePredictor(model_dir=MODEL_DIR, label_map_path=LABEL_MAP)


examples = {
    "উদাহরণ ১ - রহস্য": "প্রাচীন জমিদারবাড়িতে রাতের পর রাত অদ্ভুত শব্দ শোনা যায়। এক তরুণ তদন্তকারী হারিয়ে যাওয়া উত্তরাধিকারীর সূত্র খুঁজতে গিয়ে ভয়ংকর এক ষড়যন্ত্র আবিষ্কার করে।",
    "উদাহরণ ২ - বিজ্ঞান কল্পকাহিনি": "২১০০ সালে মঙ্গল গ্রহে বসতি স্থাপনের সময় একদল বিজ্ঞানী এমন একটি সংকেত পায় যা মানুষের ইতিহাস সম্পর্কে ভিন্ন সত্য উন্মোচন করে।",
    "উদাহরণ ৩ - জীবনী": "গ্রামের দরিদ্র পরিবারে জন্ম নেওয়া এক শিক্ষক কিভাবে সংগ্রাম করে জাতীয় শিক্ষাবিদে পরিণত হলেন, তার অনুপ্রেরণামূলক জীবনের গল্প।",
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
- Classes: 7 Bengali genre labels
- Input: Bengali Unicode summary
- Output: Predicted genre + confidence
"""
    )
    st.markdown('<p class="small-note">মডেল ফাইল না থাকলে আগে training চালান।</p>', unsafe_allow_html=True)

if predict_clicked:
    if not user_text.strip():
        st.warning("অনুগ্রহ করে একটি সারাংশ লিখুন।")
    elif not MODEL_DIR.exists() or not LABEL_MAP.exists():
        st.error("মডেল পাওয়া যায়নি। আগে preprocessing + training স্ক্রিপ্ট চালান।")
    else:
        predictor = load_predictor()
        result = predictor.predict(user_text)

        st.markdown("### Prediction")
        st.markdown(
            f"""
<div class="pred-box">
  <h3 style="margin-top:0;">{result['predicted_genre']}</h3>
  <p>Confidence: <b>{result['confidence'] * 100:.2f}%</b></p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(min(max(result["confidence"], 0.0), 1.0))

        st.markdown("### Class Probabilities")
        sorted_probs = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        for label, prob in sorted_probs:
            st.write(f"{label}: {prob * 100:.2f}%")
