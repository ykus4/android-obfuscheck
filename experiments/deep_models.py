"""Neural baselines on the same grouped folds as compare_models.py (PyTorch).

    python experiments/deep_models.py data/new_train.csv --out experiments/results/deep.json

- upstream-lstm: liansecurityOS/apk-obfucation-detection's architecture (lower-cased word
  tokens, Embedding(100k, 100) -> LSTM(128) -> sigmoid, maxlen 256, pre-padding).
- char-bilstm / char-cnn: character-level models, which see inside identifiers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from compare_models import make_folds, metrics  # noqa: E402
from features import text_plain  # noqa: E402

from android_obfuscheck.train import load_dataset  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class WordVocab:
    def __init__(self, texts, size=100_000):
        counts = Counter(t for s in texts for t in s.lower().split())
        self.index = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(size - 2))}

    def encode(self, s):
        return [self.index.get(t, 1) for t in s.lower().split()]


class CharVocab:
    def __init__(self, texts):
        chars = sorted({ch for s in texts for ch in s})
        self.index = {ch: i + 2 for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.index.get(ch, 1) for ch in s]


def pad(seqs, maxlen, pre=True):
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:] if pre else s[:maxlen]
        if pre:
            out[i, maxlen - len(s) :] = s
        else:
            out[i, : len(s)] = s
    return torch.from_numpy(out)


class UpstreamLSTM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 100, padding_idx=0)
        self.lstm = nn.LSTM(100, 128, batch_first=True)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.out(h[-1]).squeeze(-1)


class CharBiLSTM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e4)
        return self.out(h.max(dim=1).values).squeeze(-1)


class CharCNN(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(32, 128, k, padding=k // 2) for k in (3, 5, 7))
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(384, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.out(self.drop(h)).squeeze(-1)


# name: (vocab, model, maxlen, pre-pad, max epochs, early-stopping patience)
SPECS = {
    # Upstream trains up to 24 epochs with patience 5 on val accuracy.
    "upstream-lstm (word, Keras-equivalent)": (WordVocab, UpstreamLSTM, 256, True, 24, 5),
    "char-bilstm": (CharVocab, CharBiLSTM, 256, False, 12, 3),
    "char-cnn": (CharVocab, CharCNN, 256, False, 12, 3),
}


def predict_scores(model, x, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(torch.sigmoid(model(x[i : i + bs].to(DEVICE))).cpu())
    return torch.cat(out).numpy()


def fit(model, x, y, epochs=12, patience=3, bs=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_val = len(x) // 10
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x[idx].to(DEVICE)), yt[idx].to(DEVICE))
            loss.backward()
            opt.step()
        acc = float(((predict_scores(model, x[val]) >= 0.5) == y[val.numpy()]).mean())
        if acc > best + 1e-3:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="experiments/results/deep.json")
    ap.add_argument("--only", help="substring filter on model names")
    args = ap.parse_args()
    torch.manual_seed(0)

    X, labels = load_dataset(args.data)
    y = np.array(labels)
    texts = text_plain(X)
    folds = make_folds(X, y, grouped=True)
    results = {}
    for name, (vocab_cls, model_cls, maxlen, pre, epochs, patience) in SPECS.items():
        if args.only and args.only not in name:
            continue
        per_fold, fit_s, pred_ms, params = [], [], [], 0
        for tr, te in folds:
            vocab = vocab_cls([texts[i] for i in tr])
            xtr = pad([vocab.encode(texts[i]) for i in tr], maxlen, pre)
            xte = pad([vocab.encode(texts[i]) for i in te], maxlen, pre)
            size = (max(vocab.index.values()) + 1) if vocab.index else 2
            model = model_cls(size).to(DEVICE)
            params = sum(p.numel() for p in model.parameters())
            t0 = time.perf_counter()
            fit(model, xtr, y[tr], epochs=epochs, patience=patience)
            fit_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            score = predict_scores(model, xte)
            pred_ms.append((time.perf_counter() - t0) / len(te) * 1e6)
            per_fold.append(metrics(y[te], score >= 0.5, score))
        keys = per_fold[0].keys()
        r = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
        r.update({f"{k}_std": float(np.std([f[k] for f in per_fold])) for k in keys})
        r.update(
            fit_seconds=float(np.mean(fit_s)),
            predict_ms_per_1k=float(np.mean(pred_ms)),
            model_kb=params * 4 / 1024,
            device=DEVICE,
            fold_accuracy=[f["accuracy"] for f in per_fold],
        )
        results[name] = r
        print(
            f"{name:<40} acc {r['accuracy']:.4f}±{r['accuracy_std']:.4f} f1 {r['f1']:.4f} "
            f"auc {r['roc_auc']:.4f} fit {r['fit_seconds']:.0f}s",
            flush=True,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text())["results"] if out.exists() else {}
    merged.update(results)
    out.write_text(json.dumps({"results": merged}, indent=2) + "\n")


if __name__ == "__main__":
    main()
