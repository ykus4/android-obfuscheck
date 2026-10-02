"""Aggregate per-class verdicts into coverage numbers, gate them, and render."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from android_obfuscheck.classinfo import ClassInfo

# Third-party/platform code that ships unobfuscated in most apps and would drown out the signal.
DEFAULT_EXCLUDES = (
    "android.",
    "androidx.",
    "com.android.",
    "com.google.",
    "dalvik.",
    "java.",
    "javax.",
    "kotlin.",
    "kotlinx.",
    "okhttp3.",
    "okio.",
    "org.intellij.",
    "org.jetbrains.",
    "retrofit2.",
    "dagger.",
    "io.reactivex.",
    "com.squareup.",
    "com.facebook.",
)

ROOT_PACKAGE = "(root)"


def _matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(
        name == p.rstrip(".") or name.startswith(p if p.endswith(".") else p + ".")
        for p in prefixes
    )


def in_scope(c: ClassInfo, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if include and not _matches(c.name, include):
        return False
    return not _matches(c.name, exclude)


def package_key(c: ClassInfo, depth: int) -> str:
    pkg = c.package
    if not pkg:
        return ROOT_PACKAGE
    return ".".join(pkg.split(".")[:depth])


@dataclass
class PackageStats:
    package: str
    total: int = 0
    obfuscated: int = 0

    @property
    def coverage(self) -> float:
        return self.obfuscated / self.total if self.total else 0.0


@dataclass
class Report:
    target: str
    detector: str
    total_classes: int
    scoped_classes: int
    obfuscated_classes: int
    packages: list[PackageStats]
    unobfuscated: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    apkid: dict | None = None  # ApkidResult.to_dict() when --apkid was used

    @property
    def coverage(self) -> float:
        return self.obfuscated_classes / self.scoped_classes if self.scoped_classes else 0.0

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coverage"] = round(self.coverage, 4)
        d["passed"] = self.passed
        d["packages"] = [{**asdict(p), "coverage": round(p.coverage, 4)} for p in self.packages]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def build_report(
    target: str,
    detector: str,
    classes: Sequence[ClassInfo],
    verdicts: Sequence[bool],
    include: Sequence[str] = (),
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
    package_depth: int = 3,
) -> Report:
    stats: dict[str, PackageStats] = defaultdict(lambda: PackageStats(""))
    scoped = obfuscated = 0
    unobfuscated: list[str] = []
    for c, obf in zip(classes, verdicts, strict=True):
        if not in_scope(c, include, exclude):
            continue
        scoped += 1
        obfuscated += obf
        key = package_key(c, package_depth)
        s = stats[key]
        s.package = key
        s.total += 1
        s.obfuscated += obf
        if not obf:
            unobfuscated.append(c.name)
    packages = sorted(stats.values(), key=lambda p: (-p.total, p.package))
    return Report(
        target, detector, len(classes), scoped, obfuscated, packages, sorted(unobfuscated)
    )


def apply_gate(
    report: Report,
    min_coverage: float | None = None,
    baseline: dict | None = None,
    max_drop: float = 0.05,
) -> Report:
    if report.scoped_classes == 0:
        report.failures.append("no classes in scope; check --include/--exclude")
        return report
    if min_coverage is not None and report.coverage < min_coverage:
        report.failures.append(
            f"coverage {report.coverage:.1%} is below the minimum {min_coverage:.1%}"
        )
    if baseline is not None:
        base = float(baseline.get("coverage", 0.0))
        if base - report.coverage > max_drop:
            report.failures.append(
                f"coverage dropped {base - report.coverage:.1%} from baseline {base:.1%} "
                f"(allowed {max_drop:.1%})"
            )
    return report


def render_text(report: Report, top: int = 15, list_unobfuscated: int = 0) -> str:
    lines = [
        f"target:     {report.target}",
        f"detector:   {report.detector}",
        f"classes:    {report.total_classes} total, {report.scoped_classes} in scope",
        f"coverage:   {report.coverage:.1%} ({report.obfuscated_classes}/{report.scoped_classes})",
        "",
        f"{'package':<48} {'classes':>8} {'coverage':>9}",
    ]
    for p in report.packages[:top]:
        lines.append(f"{p.package:<48} {p.total:>8} {p.coverage:>9.1%}")
    if len(report.packages) > top:
        lines.append(f"... {len(report.packages) - top} more packages")
    if list_unobfuscated and report.unobfuscated:
        lines += ["", "readable classes:"]
        lines += [f"  {n}" for n in report.unobfuscated[:list_unobfuscated]]
        if len(report.unobfuscated) > list_unobfuscated:
            lines.append(f"  ... {len(report.unobfuscated) - list_unobfuscated} more")
    if report.apkid is not None:
        lines += ["", f"apkid {report.apkid['version']}:"]
        lines += [f"  {cat:<16} {', '.join(d)}" for cat, d in report.apkid["findings"].items()]
        if not report.apkid["findings"]:
            lines.append("  (no matches)")
    lines.append("")
    if report.passed:
        lines.append("PASS")
    else:
        lines += [f"FAIL: {msg}" for msg in report.failures]
    return "\n".join(lines)


def render_markdown(report: Report, top: int = 15, list_unobfuscated: int = 0) -> str:
    status = "✅ pass" if report.passed else "❌ fail"
    lines = [
        f"### Obfuscation coverage: {report.coverage:.1%} {status}",
        "",
        f"`{report.target}` · detector `{report.detector}` · "
        f"{report.scoped_classes} of {report.total_classes} classes in scope",
        "",
    ]
    lines += [f"- **{msg}**" for msg in report.failures]
    if report.failures:
        lines.append("")
    if report.apkid is not None:
        lines += ["| APKiD | findings |", "|---|---|"]
        lines += [f"| {cat} | {', '.join(d)} |" for cat, d in report.apkid["findings"].items()]
        if not report.apkid["findings"]:
            lines.append("| — | no matches |")
        lines.append("")
    lines += ["| package | classes | coverage |", "|---|---:|---:|"]
    for p in report.packages[:top]:
        lines.append(f"| `{p.package}` | {p.total} | {p.coverage:.1%} |")
    if list_unobfuscated and report.unobfuscated:
        shown = report.unobfuscated[:list_unobfuscated]
        lines += ["", "<details><summary>Readable classes</summary>", ""]
        lines += [f"- `{n}`" for n in shown]
        if len(report.unobfuscated) > len(shown):
            lines.append(f"- … {len(report.unobfuscated) - len(shown)} more")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
