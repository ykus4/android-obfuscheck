"""Render experiments/results/*.json as the Markdown table used in docs/model-experiments.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def pct(v: float) -> str:
    return f"{v * 100:.2f}"


def main() -> None:
    rows = {}
    for name in ("classic.json", "deep.json"):
        path = RESULTS / name
        if path.exists():
            rows.update(json.loads(path.read_text())["results"])
    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["accuracy"])
    print(
        "| model | accuracy % | ± | precision % | recall % | F1 | ROC-AUC "
        "| fit s | µs/class | size KB |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in ranked:
        auc = f"{r['roc_auc']:.4f}" if "roc_auc" in r else "–"
        size = f"{r['model_kb']:.0f}" if r.get("model_kb") else "–"
        print(
            f"| {name} | {pct(r['accuracy'])} | {pct(r['accuracy_std'])} | {pct(r['precision'])} "
            f"| {pct(r['recall'])} | {r['f1']:.4f} | {auc} | {r['fit_seconds']:.1f} "
            f"| {r['predict_ms_per_1k']:.1f} | {size} |"
        )


if __name__ == "__main__":
    sys.exit(main())
