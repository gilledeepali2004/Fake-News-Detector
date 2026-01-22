import streamlit as st
import joblib
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.25);
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("c.png")


st.set_page_config(page_title="Fake News Detector")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("📰 Fake News Detector")

text = st.text_area("Enter news text")

if st.button("Check"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]

        if result == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")





