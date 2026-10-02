"""Per-class obfuscation detectors.

``HeuristicDetector`` needs no model and recognises R8/ProGuard-style short names.
``MLDetector`` loads a scikit-learn pipeline produced by ``android-obfuscheck train``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from android_obfuscheck.classinfo import ClassInfo

# Names R8/ProGuard hand out: a, b, aa, Zb, a1, b12, A0, C5, o90, zze (Play Services), ...
OBFUSCATED_NAME = re.compile(
    r"^(?:[a-zA-Z]{1,2}|[a-zA-Z]{1,2}\d{1,3}|zz[a-z]{1,2}|[a-z]\d+[a-z]?)$"
)


def looks_obfuscated(name: str) -> bool:
    return bool(OBFUSCATED_NAME.match(name))


class Detector(Protocol):
    name: str

    def predict(self, classes: Sequence[ClassInfo]) -> list[bool]: ...


class HeuristicDetector:
    name = "heuristic"

    def __init__(self, member_ratio: float = 0.6) -> None:
        self.member_ratio = member_ratio

    def is_obfuscated(self, c: ClassInfo) -> bool:
        parts = c.name_parts
        if parts and (
            looks_obfuscated(parts[0])
            or (len(parts) > 1 and all(looks_obfuscated(p) or p.isdigit() for p in parts))
        ):
            return True
        members = c.member_names
        if len(members) < 2:
            return False
        hits = sum(looks_obfuscated(m) for m in members)
        return hits / len(members) >= self.member_ratio

    def predict(self, classes: Sequence[ClassInfo]) -> list[bool]:
        return [self.is_obfuscated(c) for c in classes]


def _texts(classes: Sequence[ClassInfo]) -> list[str]:
    return [c.to_text() for c in classes]


def _shape_features(classes: Sequence[ClassInfo]):
    import numpy as np

    rows = []
    for c in classes:
        members = c.member_names
        lengths = [len(m) for m in members] or [0]
        first = c.name_parts[0] if c.name_parts else ""
        rows.append(
            [
                float(looks_obfuscated(first)),
                min(len(first), 40) / 40,
                sum(looks_obfuscated(m) for m in members) / max(len(members), 1),
                min(sum(lengths) / len(lengths), 40) / 40,
                min(len(members), 100) / 100,
            ]
        )
    return np.asarray(rows, dtype=float)


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    features = FeatureUnion(
        [
            (
                "chars",
                make_pipeline(
                    FunctionTransformer(_texts),
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True
                    ),
                ),
            ),
            ("shape", FunctionTransformer(_shape_features)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]
    )


class MLDetector:
    name = "ml"

    def __init__(self, pipeline, threshold: float = 0.5, meta: dict | None = None) -> None:
        self.pipeline = pipeline
        self.threshold = threshold
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.5) -> MLDetector:
        # joblib unpickles arbitrary code: only load models you trained or trust.
        import joblib

        bundle = joblib.load(path)
        return cls(bundle["pipeline"], threshold, bundle.get("meta"))

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path)

    def predict(self, classes: Sequence[ClassInfo]) -> list[bool]:
        if not classes:
            return []
        proba = self.pipeline.predict_proba(list(classes))[:, 1]
        return [bool(p >= self.threshold) for p in proba]
