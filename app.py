import streamlit as st
import base64
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import json
import zipfile
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

@st.cache_data
def load_data():
    # Create kaggle.json from Streamlit secrets
    os.makedirs("/root/.kaggle", exist_ok=True)
    kaggle_json = {
        "username": st.secrets["kaggle"]["username"],
        "key": st.secrets["kaggle"]["key"]
    }

    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump(kaggle_json, f)

    os.chmod("/root/.kaggle/kaggle.json", 0o600)

    # Download dataset
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists("data"):
        api.dataset_download_files(
            "clmentbisaillon/fake-and-real-news-dataset",
            path="data",
            unzip=True
        )
# -------------------------
# ---------------- BACKGROUND IMAGE ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
set_bg("c.png")

# -------------------------
# Clean Text Function
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].apply(clean_text)
    return df

df = load_data()

# -------------------------
# Split Data
# -------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TF-IDF & Naive Bayes
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Title with custom style
st.markdown("""
    <h1 style='text-align: center; color: white; font-family: sans-serif;'>📰 Fake News Detector</h1>
    <p style='text-align: center; color: white; font-family: sans-serif;'>Enter a news headline or article to check if it is Real or Fake</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This app uses **Naive Bayes** and **TF-IDF vectorization** to detect fake news.
- Green ✅ = Real News  
- Red ❌ = Fake News  
- Accuracy depends on your dataset
""")
st.sidebar.markdown(f"**Dataset size:** {len(df)} news articles")
st.sidebar.markdown(f"**Model Accuracy:** {accuracy*100:.2f}%")

# Input box
st.markdown("<h3 style='color:white;'>Enter News Text:</h3>", unsafe_allow_html=True)
user_input = st.text_area("", height=150)

# Prediction button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        clean_input = clean_text(user_input)
        input_vec = vectorizer.transform([clean_input])
        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0]

        if prediction == 1:
            st.success(f"✅ REAL NEWS ({probability[1]*100:.2f}% confidence)")
        else:
            st.error(f"❌ FAKE NEWS ({probability[0]*100:.2f}% confidence)")





