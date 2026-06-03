from pathlib import Path

import joblib

from . import config
from .features import build_features, context_rows


class SectionClassifier:

    def __init__(self, model_dir=None):
        d = Path(model_dir) if model_dir else Path(config.SECTION_CLASSIFIER_DIR)
        self.rf = joblib.load(d / "rf_classifier.joblib")
        self.word_vec = joblib.load(d / "tfidf_vectorizer.joblib")
        self.char_vec = joblib.load(d / "tfidf_char_vectorizer.joblib")
        self.dl_cats = joblib.load(d / "docling_cats.joblib")
        self.dl_idx = {c: i for i, c in enumerate(self.dl_cats)}

    def classify(self, chunks):
        if not chunks:
            return []
        ordered, rows = context_rows(chunks, text_key="text")
        X = build_features(rows, self.dl_idx, self.word_vec, self.char_vec, fit=False)
        preds = self.rf.predict(X)
        return [dict(c, section_type=str(p)) for c, p in zip(ordered, preds)]
