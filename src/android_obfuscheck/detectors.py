"""Per-class obfuscation detectors.

``HeuristicDetector`` needs no model and recognises R8/ProGuard-style short names.
``MLDetector`` loads a scikit-learn pipeline produced by ``android-obfuscheck train``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from android_obfuscheck.classinfo import SPECIAL_METHODS, ClassInfo

# Selected by grouped cross-validation; see docs/model-experiments.md.
MEMBER_RATIO = 0.4
MIN_MEMBERS = 1

# Names R8/ProGuard hand out: a, b, aa, Zb, a1, b12, A0, C5, o90, and Play Services'
# za*/zb*/zz* families (zza, zae, zzcpr).
OBFUSCATED_NAME = re.compile(
    r"^(?:[a-zA-Z]{1,2}|[a-zA-Z]{1,2}\d{1,3}|(?:za|zb|zz)[a-z]{0,3}|[a-z]\d+[a-z]?)$"
)

# Members R8 must keep because the platform or a superclass calls them by name.
KNOWN_MEMBERS = frozenset(
    "run call get apply invoke accept test compare compareTo toString equals hashCode clone "
    "finalize values valueOf close onClick onCreate onDestroy onResume onPause onStart onStop "
    "onReceive onBind writeToParcel describeContents CREATOR createFromParcel newArray "
    "getInstance INSTANCE Companion serialVersionUID $VALUES".split()
)
# Compiler-generated helpers ($$ExternalSyntheticLambda0, ...) inherit short member names.
SYNTHETIC_PREFIXES = ("ExternalSynthetic", "Lambda", "lambda")
# Short by convention, not by obfuscation.
RESOURCE_CLASSES = frozenset({"R", "BuildConfig"})


def looks_obfuscated(name: str) -> bool:
    return bool(OBFUSCATED_NAME.match(name))


def repackaged(c: ClassInfo) -> bool:
    """R8 moved and renamed the class (``d4.d``, ``H1.n``): package and name both look renamed.

    The upstream dataset has no packages and few member-less classes, so a model trained on it
    misses many of these; this rule backs it up on real APKs.
    """
    parts = c.name_parts
    return (
        bool(c.package and parts)
        and all(looks_obfuscated(p) for p in c.package.split("."))
        and looks_obfuscated(parts[0])
    )


class Detector(Protocol):
    name: str

    def predict(self, classes: Sequence[ClassInfo]) -> list[bool]: ...


class HeuristicDetector:
    name = "heuristic"

    def __init__(self, member_ratio: float = MEMBER_RATIO, min_members: int = MIN_MEMBERS) -> None:
        self.member_ratio = member_ratio
        self.min_members = min_members

    def is_obfuscated(self, c: ClassInfo) -> bool:
        parts = c.name_parts
        if not parts or parts[0] in RESOURCE_CLASSES:
            return False
        if any(p.startswith(SYNTHETIC_PREFIXES) for p in parts):
            return looks_obfuscated(parts[0])
        if looks_obfuscated(parts[0]) or (len(parts) > 1 and looks_obfuscated(parts[-1])):
            return True
        if len(parts) > 1 and all(looks_obfuscated(p) or p.isdigit() for p in parts):
            return True
        members = [m for m in c.member_names if m not in KNOWN_MEMBERS]
        if len(members) < self.min_members:
            return False
        return sum(looks_obfuscated(m) for m in members) / len(members) >= self.member_ratio

    def predict(self, classes: Sequence[ClassInfo]) -> list[bool]:
        return [self.is_obfuscated(c) for c in classes]


def _texts(classes: Sequence[ClassInfo]) -> list[str]:
    return [c.to_text() for c in classes]


def _frac(xs, pred) -> float:
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


_CAMEL = re.compile(r"[a-z][A-Z]")


def _shape_features(classes: Sequence[ClassInfo]):
    import numpy as np

    heuristic = HeuristicDetector()
    rows = []
    for c in classes:
        parts = c.name_parts or [""]
        methods = [m for m in c.methods if m not in SPECIAL_METHODS]
        fields = list(c.fields)
        members = methods + fields
        unknown = [m for m in members if m not in KNOWN_MEMBERS]
        lengths = [len(m) for m in members] or [0]
        rows.append(
            [
                float(looks_obfuscated(parts[0])),
                float(looks_obfuscated(parts[-1])),
                float(all(looks_obfuscated(p) or p.isdigit() for p in parts)),
                float(parts[0] in RESOURCE_CLASSES),
                float(any(p.startswith(SYNTHETIC_PREFIXES) for p in parts)),
                min(len(parts), 5) / 5,
                min(len(parts[0]), 40) / 40,
                min(len(parts[-1]), 40) / 40,
                min(len(methods), 100) / 100,
                min(len(fields), 100) / 100,
                _frac(members, looks_obfuscated),
                _frac(unknown, looks_obfuscated),
                _frac(methods, looks_obfuscated),
                _frac(fields, looks_obfuscated),
                _frac(members, lambda m: len(m) <= 1),
                _frac(members, lambda m: len(m) <= 3),
                _frac(members, lambda m: m in KNOWN_MEMBERS),
                _frac(members, lambda m: bool(_CAMEL.search(m))),
                _frac(members, lambda m: any(ch.isdigit() for ch in m)),
                _frac(members, lambda m: "_" in m),
                _frac(members, lambda m: m[:1].isupper()),
                _frac(members, lambda m: m.startswith(("za", "zb", "zz"))),
                min(sum(lengths) / len(lengths), 40) / 40,
                min(max(lengths), 60) / 60,
                float(heuristic.is_obfuscated(c)),
            ]
        )
    return np.asarray(rows, dtype=float)


ALGORITHMS = ("lr", "lightgbm")


def build_pipeline(algorithm: str = "lr"):
    """Character n-gram TF-IDF plus shape features, then a classifier.

    ``lr`` (logistic regression) is the default: within 0.1 pt of LightGBM in grouped CV with no
    extra dependency. ``lightgbm`` needs ``pip install lightgbm``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    if algorithm == "lr":
        from sklearn.linear_model import LogisticRegression

        ngrams = (1, 4)
        clf = LogisticRegression(max_iter=3000, C=16.0, class_weight="balanced")
    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        ngrams = (1, 3)
        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            colsample_bytree=0.5,
            class_weight="balanced",
            random_state=0,
            verbose=-1,
        )
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {', '.join(ALGORITHMS)}")

    features = FeatureUnion(
        [
            (
                "chars",
                make_pipeline(
                    FunctionTransformer(_texts),
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=ngrams, min_df=2, sublinear_tf=True
                    ),
                ),
            ),
            ("shape", FunctionTransformer(_shape_features)),
        ]
    )
    return Pipeline([("features", features), ("clf", clf)])


class MLDetector:
    name = "ml"

    def __init__(
        self,
        pipeline,
        threshold: float = 0.5,
        meta: dict | None = None,
        package_rule: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.threshold = threshold
        self.meta = meta or {}
        self.package_rule = package_rule

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
        return [
            bool(p >= self.threshold) or (self.package_rule and repackaged(c))
            for c, p in zip(classes, proba, strict=True)
        ]
