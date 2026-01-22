import streamlit as st
import joblib
import re

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- Background Image ----------------
st.markdown("""
<style>
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: url("https://images.unsplash.com/photo-1504711434969-e33886168f5c") no-repeat;
    background-size: cover;
    opacity: 0.25;  /* fade effect */
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Text Cleaning Function ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- App Title ----------------
st.markdown("<h1 style='text-align:center;color:white;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:white;'>Enter a news headline or article below to check if it is Real or Fake</p>", unsafe_allow_html=True)

# ---------------- User Input ----------------
text = st.text_area("News Text:", height=200)

# ---------------- Prediction ----------------
if st.button("Check News"):
    if text.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        clean_input = clean_text(text)
        vec = vectorizer.transform([clean_input])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        if prediction == 1:
            st.success(f"✅ REAL NEWS ({probability[1]*100:.2f}% confidence)")
        else:
            st.error(f"❌ FAKE NEWS ({probability[0]*100:.2f}% confidence)")

# ---------------- Sidebar ----------------
st.sidebar.header("About")
st.sidebar.info("""
This app uses **Naive Bayes** + **TF-IDF vectorization** for fake news detection.

- ✅ Real news  
- ❌ Fake news  
- Confidence percentage shows model certainty
""")
")






