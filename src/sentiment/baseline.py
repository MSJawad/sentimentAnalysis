"""TF-IDF over word unigrams and bigrams, fed to logistic regression."""

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment.data import Review, tokenize


class Baseline:
    def __init__(self, pipeline: Pipeline, train_seconds: float):
        self.pipeline = pipeline
        self.train_seconds = train_seconds

    @classmethod
    def fit(cls, reviews: list[Review], c: float = 4.0) -> "Baseline":
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize,
                        lowercase=False,
                        token_pattern=None,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("logreg", LogisticRegression(C=c, max_iter=1000)),
            ]
        )
        start = time.perf_counter()
        pipeline.fit([r.text for r in reviews], [r.label for r in reviews])
        return cls(pipeline, time.perf_counter() - start)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability that each text is positive."""
        return self.pipeline.predict_proba(texts)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "train_seconds": self.train_seconds}, path)

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        saved = joblib.load(path)
        return cls(saved["pipeline"], saved["train_seconds"])
