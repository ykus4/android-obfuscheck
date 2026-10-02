"""Compare obfuscation detectors with grouped, stratified 5-fold cross-validation.

    python experiments/compare_models.py data/new_train.csv --out experiments/results/classic.json

Rows with identical text (``R``, ``a a`` ... recur across apps) are kept in the same fold so
that duplicates cannot leak from train to test. Results feed docs/model-experiments.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402

from android_obfuscheck.train import load_dataset  # noqa: E402


def tfidf(text_fn, ngram=(1, 3), analyzer="char_wb", min_df=2):
    return make_pipeline(
        FunctionTransformer(text_fn),
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, sublinear_tf=True),
    )


def union(text_fn, shape_fn, ngram=(1, 3)):
    return FeatureUnion([("text", tfidf(text_fn, ngram)), ("shape", FunctionTransformer(shape_fn))])


def lr(C=4.0):
    return LogisticRegression(max_iter=3000, C=C, class_weight="balanced")


def lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        colsample_bytree=0.5,
        class_weight="balanced",
        verbose=-1,
        random_state=0,
    )


def candidates():
    shape2 = FunctionTransformer(F.shape_v2)
    lr_v2 = Pipeline([("f", union(F.text_plain, F.shape_v2, (1, 4))), ("clf", lr())])
    hgb = Pipeline(
        [("f", shape2), ("clf", HistGradientBoostingClassifier(max_iter=300, random_state=0))]
    )
    lgb = Pipeline([("f", union(F.text_plain, F.shape_v2, (1, 3))), ("clf", lgbm())])
    return {
        "lr-v1 (v0.1 model)": Pipeline([("f", union(F.text_plain, F.shape_v1)), ("clf", lr())]),
        "lr char 1-5 + shape-v1": Pipeline(
            [("f", union(F.text_plain, F.shape_v1, (1, 5))), ("clf", lr())]
        ),
        "lr tagged 1-4 + shape-v1": Pipeline(
            [("f", union(F.text_tagged, F.shape_v1, (1, 4))), ("clf", lr())]
        ),
        "lr char 1-4 + shape-v2": lr_v2,
        "lr char 1-4 + shape-v2 C=1": Pipeline(
            [("f", union(F.text_plain, F.shape_v2, (1, 4))), ("clf", lr(1.0))]
        ),
        "lr char 1-4 + shape-v2 C=16": Pipeline(
            [("f", union(F.text_plain, F.shape_v2, (1, 4))), ("clf", lr(16.0))]
        ),
        "lr tagged 1-4 + shape-v2": Pipeline(
            [("f", union(F.text_tagged, F.shape_v2, (1, 4))), ("clf", lr())]
        ),
        "linear svm char 1-4 + shape-v2": Pipeline(
            [
                ("f", union(F.text_plain, F.shape_v2, (1, 4))),
                ("clf", LinearSVC(C=0.5, class_weight="balanced")),
            ]
        ),
        "complement nb char 1-4": Pipeline(
            [("f", tfidf(F.text_plain, (1, 4))), ("clf", ComplementNB(alpha=0.3))]
        ),
        "hist gbdt shape-v2 only": hgb,
        "lightgbm char 1-3 + shape-v2": lgb,
        "soft vote (lr-v2 + hgb + lgbm)": VotingClassifier(
            [("lr", lr_v2), ("hgb", hgb), ("lgb", lgb)], voting="soft"
        ),
    }


HEURISTIC_GRID = [(r, m) for r in (0.4, 0.5, 0.6, 0.7) for m in (1, 2, 3)]


def metrics(y, pred, score=None):
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if score is not None:
        out["roc_auc"] = roc_auc_score(y, score)
    return out


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(name, make, X, y, folds):
    per_fold, fit_s, pred_ms, size_kb, extra = [], [], [], None, {}
    for i, (tr, te) in enumerate(folds):
        Xtr, Xte = [X[j] for j in tr], [X[j] for j in te]
        ytr, yte = y[tr], y[te]
        if name.startswith("heuristic-v2"):
            t0 = time.perf_counter()
            best = max(
                HEURISTIC_GRID,
                key=lambda p: np.mean(
                    [F.heuristic_v2(c, *p) == t for c, t in zip(Xtr, ytr, strict=True)]
                ),
            )
            fit_s.append(time.perf_counter() - t0)
            extra.setdefault("selected_params", []).append(best)
            t0 = time.perf_counter()
            pred = np.array([F.heuristic_v2(c, *best) for c in Xte])
            pred_ms.append((time.perf_counter() - t0) / len(Xte) * 1e6)
            per_fold.append(metrics(yte, pred))
            continue
        if name.startswith("heuristic-v1"):
            t0 = time.perf_counter()
            pred = np.array([F.heuristic_v1(c) for c in Xte])
            pred_ms.append((time.perf_counter() - t0) / len(Xte) * 1e6)
            fit_s.append(0.0)
            per_fold.append(metrics(yte, pred))
            continue
        model = make()
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pred = model.predict(Xte)
        score = score_of(model, Xte)
        pred_ms.append((time.perf_counter() - t0) / len(Xte) * 1e6)
        per_fold.append(metrics(yte, pred, score))
        if i == 0:
            size_kb = len(pickle.dumps(model)) / 1024
    keys = per_fold[0].keys()
    summary = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    summary.update({f"{k}_std": float(np.std([f[k] for f in per_fold])) for k in keys})
    summary.update(
        fit_seconds=float(np.mean(fit_s)),
        predict_ms_per_1k=float(np.mean(pred_ms)),  # µs per row == ms per 1k rows
        model_kb=size_kb,
        **extra,
    )
    return summary


def make_folds(X, y, grouped: bool, seed: int = 0):
    if grouped:
        texts = [c.to_text() for c in X]
        ids = {t: i for i, t in enumerate(dict.fromkeys(texts))}
        groups = np.array([ids[t] for t in texts])
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="experiments/results/classic.json")
    ap.add_argument("--only", help="substring filter on candidate names")
    args = ap.parse_args()

    X, labels = load_dataset(args.data)
    y = np.array(labels)
    folds = make_folds(X, y, grouped=True)
    names = ["heuristic-v1 (v0.1 default)", "heuristic-v2"] + list(candidates())
    results = {}
    for name in names:
        if args.only and args.only not in name:
            continue
        r = evaluate(name, lambda name=name: candidates()[name], X, y, folds)
        results[name] = r
        print(
            f"{name:<36} acc {r['accuracy']:.4f}±{r['accuracy_std']:.4f} "
            f"f1 {r['f1']:.4f} auc {r.get('roc_auc', float('nan')):.4f} "
            f"fit {r['fit_seconds']:.1f}s",
            flush=True,
        )

    # The number v0.1's README reported came from an ungrouped split; show the difference.
    if not args.only:
        ungrouped = make_folds(X, y, grouped=False)
        r = evaluate("lr-v1", lambda: candidates()["lr-v1 (v0.1 model)"], X, y, ungrouped)
        results["lr-v1 (v0.1 model), ungrouped folds"] = r
        print(f"{'lr-v1 ungrouped':<36} acc {r['accuracy']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "rows": len(y),
        "positives": int(y.sum()),
        "unique_texts": len({c.to_text() for c in X}),
        "cv": "StratifiedGroupKFold(5, shuffle, seed=0), groups = identical text",
    }
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
