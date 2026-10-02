"""Candidate detectors and feature sets compared in docs/model-experiments.md.

Kept separate from the package so that baselines (v0.1 heuristic, v0.1 pipeline) stay
reproducible after the package moves on.
"""

from __future__ import annotations

import re

import numpy as np

from android_obfuscheck.classinfo import ClassInfo

# --- v0.1 (as released) -------------------------------------------------------------------

OBF_V1 = re.compile(r"^(?:[a-zA-Z]{1,2}|[a-zA-Z]{1,2}\d{1,3}|zz[a-z]{1,2}|[a-z]\d+[a-z]?)$")


def heuristic_v1(c: ClassInfo, member_ratio: float = 0.6) -> bool:
    parts = c.name_parts
    if parts and (
        OBF_V1.match(parts[0])
        or (len(parts) > 1 and all(OBF_V1.match(p) or p.isdigit() for p in parts))
    ):
        return True
    members = c.member_names
    if len(members) < 2:
        return False
    return sum(bool(OBF_V1.match(m)) for m in members) / len(members) >= member_ratio


def shape_v1(classes):
    rows = []
    for c in classes:
        members = c.member_names
        lengths = [len(m) for m in members] or [0]
        first = c.name_parts[0] if c.name_parts else ""
        rows.append(
            [
                float(bool(OBF_V1.match(first))),
                min(len(first), 40) / 40,
                sum(bool(OBF_V1.match(m)) for m in members) / max(len(members), 1),
                min(sum(lengths) / len(lengths), 40) / 40,
                min(len(members), 100) / 100,
            ]
        )
    return np.asarray(rows, dtype=float)


# --- v2 candidates ------------------------------------------------------------------------

# Adds Play Services' za*/zb*/zz* families (zza, zae, zzcpr).
OBF_V2 = re.compile(
    r"^(?:[a-zA-Z]{1,2}|[a-zA-Z]{1,2}\d{1,3}|(?:za|zb|zz)[a-z]{0,3}|[a-z]\d+[a-z]?)$"
)

# Members R8 must keep because the platform or a superclass calls them by name.
KNOWN_MEMBERS = frozenset(
    "run call get apply invoke accept test compare compareTo toString equals hashCode clone "
    "finalize values valueOf close onClick onCreate onDestroy onResume onPause onStart onStop "
    "onReceive onBind writeToParcel describeContents CREATOR createFromParcel newArray "
    "getInstance INSTANCE Companion serialVersionUID $VALUES".split()
)
SYNTHETIC_PREFIXES = ("ExternalSynthetic", "Lambda", "lambda")
RESOURCE_CLASSES = frozenset({"R", "BuildConfig"})


def _obf(name: str) -> bool:
    return bool(OBF_V2.match(name))


def heuristic_v2(c: ClassInfo, member_ratio: float = 0.5, min_members: int = 2) -> bool:
    parts = c.name_parts
    if not parts or parts[0] in RESOURCE_CLASSES:
        return False
    if any(p.startswith(SYNTHETIC_PREFIXES) for p in parts):
        # Synthetic helpers inherit short member names; judge by the outer class only.
        return _obf(parts[0])
    if _obf(parts[0]) or (len(parts) > 1 and _obf(parts[-1])):
        return True
    if len(parts) > 1 and all(_obf(p) or p.isdigit() for p in parts):
        return True
    members = [m for m in c.member_names if m not in KNOWN_MEMBERS]
    if len(members) < min_members:
        return False
    return sum(_obf(m) for m in members) / len(members) >= member_ratio


def _frac(xs, pred) -> float:
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


_CAMEL = re.compile(r"[a-z][A-Z]")


def shape_v2(classes):
    rows = []
    for c in classes:
        parts = c.name_parts or [""]
        methods = [m for m in c.methods if m not in ("<init>", "<clinit>")]
        fields = list(c.fields)
        members = methods + fields
        unknown = [m for m in members if m not in KNOWN_MEMBERS]
        lengths = [len(m) for m in members] or [0]
        rows.append(
            [
                float(_obf(parts[0])),
                float(_obf(parts[-1])),
                float(all(_obf(p) or p.isdigit() for p in parts)),
                float(parts[0] in RESOURCE_CLASSES),
                float(any(p.startswith(SYNTHETIC_PREFIXES) for p in parts)),
                min(len(parts), 5) / 5,
                min(len(parts[0]), 40) / 40,
                min(len(parts[-1]), 40) / 40,
                min(len(methods), 100) / 100,
                min(len(fields), 100) / 100,
                _frac(members, _obf),
                _frac(unknown, _obf),
                _frac(methods, _obf),
                _frac(fields, _obf),
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
                float(heuristic_v2(c)),
            ]
        )
    return np.asarray(rows, dtype=float)


def text_plain(classes) -> list[str]:
    return [c.to_text() for c in classes]


def text_tagged(classes) -> list[str]:
    """Tokens prefixed by role so 'a' as a class name differs from 'a' as a field."""
    out = []
    for c in classes:
        toks = ["C" + p for p in c.name_parts]
        toks += ["M" + m for m in c.methods if m not in ("<init>", "<clinit>")]
        toks += ["F" + f for f in c.fields]
        out.append(" ".join(toks))
    return out
