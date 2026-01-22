import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load CSV files (local)
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df[["text", "label"]].dropna()

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.6,
    min_df=5,
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully")
