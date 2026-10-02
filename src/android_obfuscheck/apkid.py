"""Optional APKiD integration (https://github.com/rednaga/APKiD).

APKiD fingerprints the dex compiler (r8, d8, dx, dexlib), commercial obfuscators, packers and
anti-analysis tricks with YARA rules. That complements the name-based coverage: it tells you
*which* toolchain produced the build and whether it was rebuilt by apktool/smali (dexlib).

APKiD is GPL-3.0, so it is run as a separate executable rather than imported.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class ApkidError(Exception):
    pass


@dataclass
class ApkidResult:
    version: str = ""
    # category (compiler, obfuscator, packer, anti_vm, ...) -> sorted unique descriptions
    findings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def compilers(self) -> list[str]:
        return self.findings.get("compiler", [])

    def to_dict(self) -> dict:
        return {"version": self.version, "findings": self.findings}


def parse_apkid_json(data: dict) -> ApkidResult:
    """Merge per-file matches into one category -> descriptions map."""
    merged: dict[str, set[str]] = {}
    for entry in data.get("files", []):
        for tags, descriptions in entry.get("matches", {}).items():
            for tag in (t.strip() for t in tags.split(",")):
                if tag:
                    merged.setdefault(tag, set()).update(descriptions)
    return ApkidResult(
        version=str(data.get("apkid_version", "")),
        findings={k: sorted(v) for k, v in sorted(merged.items())},
    )


def run_apkid(path: str | Path, executable: str = "apkid", timeout: int = 600) -> ApkidResult:
    exe = shutil.which(executable)
    if exe is None:
        raise ApkidError(f"{executable!r} not found; install it with `pip install apkid`")
    try:
        proc = subprocess.run(
            [exe, "-j", str(path)], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise ApkidError(f"apkid timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise ApkidError(f"apkid exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    try:
        return parse_apkid_json(json.loads(proc.stdout))
    except json.JSONDecodeError as e:
        raise ApkidError(f"could not parse apkid output: {e}") from e


def gate_apkid(result: ApkidResult, require_compiler: list[str], forbid: list[str]) -> list[str]:
    failures = []
    if require_compiler:
        wanted = [w.lower() for w in require_compiler]
        if not any(c.lower().startswith(w) for c in result.compilers for w in wanted):
            found = ", ".join(result.compilers) or "unknown"
            failures.append(f"compiler is {found}; expected one of {', '.join(require_compiler)}")
    for category in forbid:
        hits = result.findings.get(category)
        if hits:
            failures.append(f"APKiD found {category}: {', '.join(hits)}")
    return failures
