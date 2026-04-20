import pickle
import pandas as pd
import spacy
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer

# Bind dynamically mapped math matrices out of Notebook module
from notebook_features import extract_all_features_native

# -----------------------------
# GLOBALS & MODELS
# -----------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner"])
sbert = SentenceTransformer("all-MiniLM-L6-v2")
spell = SpellChecker()

# Load the recorded explicit column arrays sequentially
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "legacy_weights/xgb_ai_detector.pkl"), "rb"))
feature_columns = pickle.load(open(os.path.join(BASE_DIR, "legacy_weights/feature_columns.pkl"), "rb"))

# -----------------------------
# MAIN FEATURE EXTRACTION
# -----------------------------
def extract_features(text):
    # Route straight into the precise 114 pipeline seamlessly
    feats = extract_all_features_native(text, nlp=nlp, sbert=sbert, spell=spell)
    for col in feature_columns:
        if col not in feats:
            feats[col] = 0.0
    return pd.DataFrame([feats], columns=feature_columns)

# -----------------------------
# PREDICT
# -----------------------------
def predict(text):
    X = extract_features(text)

    # Utilizing completely anonymous values array bypasses pandas fragmentation mismatches!
    prob = model.predict_proba(X.values)[0][1]
    pred = model.predict(X.values)[0]

    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }

# -----------------------------
# CLI INTERFACE 
# -----------------------------
if __name__ == "__main__":
    text = input("Enter text string:\n")
    res = predict(text)

    print("\nPrediction Evaluated:", "AI-generated Patterning" if res["prediction"] == 1 else "Human-written Patterning")
    print("Probability Confidence Vector:", res["confidence"])