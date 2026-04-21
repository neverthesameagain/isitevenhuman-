"""
AI Text Detector — Production Web Application (v2)
Supports:
  • Hot-swapping between NEW and OLD model weights
  • Selective feature-group toggling for prediction
"""

import os
import pickle
import numpy as np
import pandas as pd
import spacy
from collections import OrderedDict
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "legacy_weights")

MODEL_REGISTRY = {
    "new": {
        "model": os.path.join(WEIGHTS_DIR, "xgb_ai_detector.pkl"),
        "columns": os.path.join(WEIGHTS_DIR, "feature_columns.pkl"),
    },
    "old": {
        "model": os.path.join(WEIGHTS_DIR, "old_xgb_ai_detector.pkl"),
        "columns": os.path.join(WEIGHTS_DIR, "old_feature_columns.pkl"),
    },
}

# ── Feature group definitions (maps directly to notebook cells) ─────
FEATURE_GROUPS = OrderedDict({
    "sentence": {
        "label": "Sentence Structure",
        "description": "Length stats, burstiness, coefficient of variation",
        "prefixes": ["sent_len_"],
        "keys": ["burstiness"],
    },
    "lexical": {
        "label": "Lexical Diversity",
        "description": "TTR variants, hapax ratio, vocabulary richness",
        "prefixes": ["ttr", "root_ttr", "corrected_ttr", "log_ttr", "mtld_proxy", "msttr", "hapax_ratio", "dis_ratio", "rare_ratio", "freq_entropy", "gini_coef", "top10_coverage", "top1_word_freq", "vocab_size"],
        "keys": [],
    },
    "marker": {
        "label": "Discourse Markers",
        "description": "Template phrases, safe vocab, importance words",
        "prefixes": ["discourse_", "safe_vocab", "importance_", "template_", "it_is_", "this_is_", "there_is_", "sentence_start_marker", "synonym_cycle", "intro_template", "ending_template", "stacked_marker", "marker_burstiness"],
        "keys": [],
    },
    "entropy": {
        "label": "Entropy & Repetition",
        "description": "Shannon entropy, n-gram reuse, local entropy variation",
        "prefixes": ["word_entropy", "normalized_word", "bigram_entropy", "normalized_bigram", "trigram_entropy", "normalized_trigram", "bigram_reuse", "trigram_reuse", "fourgram_reuse", "word_redundancy", "long_phrase", "local_entropy", "entropy_variation", "zipf_deviation"],
        "keys": [],
    },
    "zipf": {
        "label": "Zipf's Law",
        "description": "Frequency rank distribution shape and deviation",
        "prefixes": ["zipf_"],
        "keys": ["hapax_ratio", "dis_ratio", "top_10_token", "top_50_token", "vocab_token_ratio"],
    },
    "pos": {
        "label": "POS Tagging",
        "description": "Part-of-speech ratios, syntactic transitions",
        "prefixes": ["pos_", "conj_rate", "aux_rate", "noun_verb", "content_function", "transition_", "sentence_length_std_pos", "noun_rate_sentence", "verb_rate_sentence", "modifier_stack"],
        "keys": [],
    },
    "informality": {
        "label": "Informality",
        "description": "Typos, contractions, slang, personal pronouns",
        "prefixes": ["typo_", "contraction_", "informal_", "first_person_", "second_person_", "question_rate", "exclamation_rate", "em_dash_", "ellipsis_"],
        "keys": [],
    },
    "structural": {
        "label": "Structural Layout",
        "description": "Paragraph stats, list detection, headers",
        "prefixes": ["para_", "list_", "line_len", "line_uniformity", "template_count", "header_count", "intro_outro_balance"],
        "keys": [],
    },
    "sbert": {
        "label": "SBERT Coherence",
        "description": "Sentence embedding similarity and coherence flow",
        "prefixes": ["sbert_"],
        "keys": [],
    },
})


def _feature_belongs_to_group(feature_name: str, group_key: str) -> bool:
    """Check if a feature column belongs to a specific group."""
    info = FEATURE_GROUPS[group_key]
    if feature_name in info.get("keys", []):
        return True
    for prefix in info.get("prefixes", []):
        if feature_name.startswith(prefix):
            return True
    return False


def _get_group_for_feature(feature_name: str) -> str:
    """Return the group key a feature belongs to, or 'other'."""
    for g in FEATURE_GROUPS:
        if _feature_belongs_to_group(feature_name, g):
            return g
    return "other"


# ── heavy model singletons ──────────────────────────────────────────
print("Loading NLP models …")
nlp = None
sbert = None
spell = None

def load_models():
    global nlp, sbert, spell
    if nlp is None:
        print("Loading spaCy...")
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    if sbert is None:
        print("Loading SBERT...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    if spell is None:
        spell = SpellChecker()
print("NLP models ready.")

from notebook_features import extract_all_features_native

# ── XGBoost weight cache ────────────────────────────────────────────
_cache: dict = {}


def _load_weights(variant: str):
    if variant in _cache:
        return _cache[variant]
    paths = MODEL_REGISTRY[variant]
    model = pickle.load(open(paths["model"], "rb"))
    cols = pickle.load(open(paths["columns"], "rb"))
    _cache[variant] = (model, cols)
    return model, cols


for v in MODEL_REGISTRY:
    _load_weights(v)
print("XGBoost weights loaded (new + old).")


# ── feature extraction ──────────────────────────────────────────────
def build_feature_vector(text: str, feature_columns, enabled_groups: Optional[List[str]] = None):
    """Extract features; zero-out any groups not in enabled_groups."""
    feats = extract_all_features_native(text, nlp=nlp, sbert=sbert, spell=spell)

    for col in feature_columns:
        if col not in feats:
            feats[col] = 0.0

    # If specific groups selected, zero-out features outside those groups
    if enabled_groups is not None:
        enabled_set = set(enabled_groups)
        for col in feature_columns:
            group = _get_group_for_feature(col)
            if group not in enabled_set:
                feats[col] = 0.0

    return pd.DataFrame([feats], columns=feature_columns), feats


# ── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(title="AI Text Detector v2")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


class PredictRequest(BaseModel):
    text: str
    model_variant: str = "new"
    enabled_groups: Optional[List[str]] = None  # None = all enabled


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/api/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys())}


@app.get("/api/feature-groups")
def list_feature_groups():
    """Return available feature groups with metadata."""
    groups = []
    for key, info in FEATURE_GROUPS.items():
        groups.append({
            "key": key,
            "label": info["label"],
            "description": info["description"],
        })
    return {"groups": groups}


from fastapi import HTTPException
from collections import OrderedDict
import numpy as np

@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        # ── Load models ──
        load_models()

        # ── Input validation ──
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        if req.model_variant not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown model variant: {req.model_variant}")

        # ── Load weights ──
        model, feature_columns = _load_weights(req.model_variant)

        # ── Overall prediction ──
        X, raw_feats = build_feature_vector(req.text, feature_columns, req.enabled_groups)
        X_np = X.values.astype(np.float64)

        prob = float(model.predict_proba(X_np)[0][1])
        pred = int(model.predict(X_np)[0])

        # ── Feature annotation ──
        feature_details = OrderedDict()
        for col in feature_columns:
            try:
                val = raw_feats.get(col, 0.0)

                if hasattr(val, "item"):
                    val = val.item()

                group = _get_group_for_feature(col)
                is_active = req.enabled_groups is None or group in req.enabled_groups

                feature_details[col] = {
                    "value": round(float(val), 6),
                    "group": group,
                    "active": is_active,
                }

            except Exception as e:
                print(f"[Feature Error] {col}: {str(e)}")  # 🔍 debug
                feature_details[col] = {
                    "value": 0.0,
                    "group": "unknown",
                    "active": False,
                }

        # ── Final response ──
        return {
            "prediction": "AI-generated" if pred == 1 else "Human-written",
            "probability_ai": round(prob, 4),
            "probability_human": round(1 - prob, 4),
            "model_variant": req.model_variant,
            "feature_count": len(feature_columns),
            "active_features": sum(1 for f in feature_details.values() if f["active"]),
            "features": feature_details,
        }

    except HTTPException as e:
        # Already clean error
        raise e

    except Exception as e:
        # 🔥 THIS IS THE IMPORTANT PART
        print("[FATAL ERROR]:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
