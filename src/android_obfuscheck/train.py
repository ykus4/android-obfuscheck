"""Train an ``MLDetector`` from a ``text,label`` CSV (the upstream new_train.csv format)."""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from android_obfuscheck.classinfo import ClassInfo, parse_upstream_text
from android_obfuscheck.detectors import HeuristicDetector, MLDetector, build_pipeline


def load_dataset(path: str | Path) -> tuple[list[ClassInfo], list[int]]:
    csv.field_size_limit(sys.maxsize)
    classes, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if label not in ("0", "1"):
                continue
            info = parse_upstream_text(row.get("text") or "")
            if info is None:
                continue
            classes.append(info)
            labels.append(int(label))
    return classes, labels


def _metrics(y_true: list[int], y_pred: list[bool]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def grouped_split(classes: list[ClassInfo], labels: list[int], test_size: float, seed: int):
    """Stratified hold-out that keeps identical class texts on the same side.

    The upstream dataset repeats classes such as ``R`` or ``a a`` across apps; a plain random
    split would score those duplicates as if they were unseen.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    texts = [c.to_text() for c in classes]
    ids = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    n_splits = max(2, round(1 / test_size))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(cv.split(texts, labels, [ids[t] for t in texts]))
    return (
        [classes[i] for i in tr],
        [classes[i] for i in te],
        [labels[i] for i in tr],
        [labels[i] for i in te],
    )


def train(
    data: str | Path,
    out: str | Path,
    test_size: float = 0.2,
    seed: int = 0,
    algorithm: str = "lr",
) -> dict:
    classes, labels = load_dataset(data)
    if len(set(labels)) < 2:
        raise ValueError(f"{data}: need both label 0 and label 1 rows, got {len(labels)} rows")

    x_tr, x_te, y_tr, y_te = grouped_split(classes, labels, test_size, seed)
    pipeline = build_pipeline(algorithm).fit(x_tr, y_tr)
    evaluation = {
        "ml": _metrics(y_te, MLDetector(pipeline).predict(x_te)),
        "heuristic": _metrics(y_te, HeuristicDetector().predict(x_te)),
    }

    # Refit on everything for the shipped model; the held-out numbers above stay honest.
    pipeline = build_pipeline(algorithm).fit(classes, labels)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": Path(data).name,
        "rows": len(labels),
        "algorithm": algorithm,
        "test_size": test_size,
        "seed": seed,
        "split": "stratified, grouped by identical text",
        "evaluation": evaluation,
    }
    MLDetector(pipeline, meta=meta).save(out)
    return meta
