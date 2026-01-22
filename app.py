import streamlit as st
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# ---------------- Load Model & Vectorizer ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- Calculate Accuracy ----------------
# Load dataset again for testing accuracy
import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- Sidebar ----------------
st.sidebar.header("About")
st.sidebar.info("""
This app uses **Naive Bayes** + **TF-IDF vectorization** for fake news detection.

- ✅ Real news  
- ❌ Fake news  
- Confidence percentage shows model certainty
""")
st.sidebar.markdown(f"**Model Accuracy:** {accuracy*100:.2f}%")

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






