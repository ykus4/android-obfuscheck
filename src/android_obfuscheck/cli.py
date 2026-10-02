"""Command line entry point: ``android-obfuscheck scan`` and ``android-obfuscheck train``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from android_obfuscheck import __version__
from android_obfuscheck.report import (
    DEFAULT_EXCLUDES,
    apply_gate,
    build_report,
    render_markdown,
    render_text,
)

EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2


def _load_detector(args: argparse.Namespace):
    from android_obfuscheck.detectors import HeuristicDetector, MLDetector

    if args.model:
        try:
            return MLDetector.load(args.model, threshold=args.threshold)
        except ImportError:
            sys.exit("--model needs scikit-learn: pip install 'android-obfuscheck[ml]'")
    return HeuristicDetector()


def cmd_scan(args: argparse.Namespace) -> int:
    from android_obfuscheck.extract import ExtractionError, extract_classes

    try:
        classes = extract_classes(args.target)
    except (OSError, ExtractionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR

    detector = _load_detector(args)
    exclude = list(args.exclude) + ([] if args.no_default_excludes else list(DEFAULT_EXCLUDES))
    report = build_report(
        target=Path(args.target).name,
        detector=detector.name,
        classes=classes,
        verdicts=detector.predict(classes),
        include=args.include,
        exclude=exclude,
        package_depth=args.package_depth,
    )
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    apply_gate(report, args.min_coverage, baseline, args.max_drop)

    if args.apkid or args.require_compiler or args.forbid:
        from android_obfuscheck.apkid import ApkidError, gate_apkid, run_apkid

        try:
            result = run_apkid(args.target, args.apkid_bin)
        except ApkidError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_ERROR
        report.apkid = result.to_dict()
        report.failures += gate_apkid(result, args.require_compiler, args.forbid)

    if args.json_out:
        Path(args.json_out).write_text(report.to_json() + "\n")
    if args.markdown_out:
        with open(args.markdown_out, "a", encoding="utf-8") as f:
            f.write(render_markdown(report, args.top, args.list_unobfuscated))

    if args.format == "json":
        print(report.to_json())
    elif args.format == "markdown":
        print(render_markdown(report, args.top, args.list_unobfuscated), end="")
    else:
        print(render_text(report, args.top, args.list_unobfuscated))
    return EXIT_OK if report.passed else EXIT_GATE_FAILED


def cmd_train(args: argparse.Namespace) -> int:
    try:
        from android_obfuscheck.train import train
    except ImportError:
        sys.exit("training needs scikit-learn: pip install 'android-obfuscheck[ml]'")
    meta = train(
        args.data, args.out, test_size=args.test_size, seed=args.seed, algorithm=args.algorithm
    )
    print(json.dumps(meta, indent=2))
    print(f"saved {args.out}", file=sys.stderr)
    return EXIT_OK


def _fraction(value: str) -> float:
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="android-obfuscheck",
        description="Measure R8/ProGuard obfuscation coverage of Android builds.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="measure obfuscation coverage of an APK/AAB/dex")
    scan.add_argument("target", help="path to .apk, .aab, .jar/.zip containing dex, or .dex")
    scan.add_argument("--model", help="model from `android-obfuscheck train` (default: heuristic)")
    scan.add_argument(
        "--threshold",
        type=_fraction,
        default=0.5,
        help="ML probability above which a class counts as obfuscated",
    )
    scan.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PREFIX",
        help="only count classes under this package (repeatable)",
    )
    scan.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PREFIX",
        help="skip classes under this package (repeatable)",
    )
    scan.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="do not skip androidx/kotlin/com.google/... by default",
    )
    scan.add_argument(
        "--package-depth", type=int, default=3, help="package segments used for the breakdown table"
    )
    scan.add_argument("--min-coverage", type=_fraction, help="fail if coverage is below this")
    scan.add_argument("--baseline", help="JSON report from a previous run to compare against")
    scan.add_argument(
        "--max-drop", type=_fraction, default=0.05, help="allowed coverage drop versus --baseline"
    )
    scan.add_argument(
        "--apkid",
        action="store_true",
        help="also fingerprint compiler/packer/obfuscator with APKiD",
    )
    scan.add_argument("--apkid-bin", default="apkid", help="APKiD executable")
    scan.add_argument(
        "--require-compiler",
        action="append",
        default=[],
        metavar="NAME",
        help="fail unless APKiD reports this compiler, e.g. r8 (implies --apkid)",
    )
    scan.add_argument(
        "--forbid",
        action="append",
        default=[],
        metavar="CATEGORY",
        help="fail if APKiD reports this category, e.g. packer, manipulator (implies --apkid)",
    )
    scan.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    scan.add_argument("--json-out", help="also write the JSON report here")
    scan.add_argument(
        "--markdown-out", help="append a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)"
    )
    scan.add_argument("--top", type=int, default=15, help="packages shown in the table")
    scan.add_argument(
        "--list-unobfuscated",
        type=int,
        default=0,
        metavar="N",
        help="list up to N in-scope classes that stayed readable",
    )
    scan.set_defaults(func=cmd_scan)

    tr = sub.add_parser("train", help="train an ML model from a text,label CSV")
    tr.add_argument("--data", required=True, help="CSV in the new_train.csv format")
    tr.add_argument("--out", default="android-obfuscheck-model.joblib")
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument(
        "--algorithm",
        choices=["lr", "lightgbm"],
        default="lr",
        help="classifier; lightgbm is ~0.1 pt more accurate but needs `pip install lightgbm`",
    )
    tr.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
