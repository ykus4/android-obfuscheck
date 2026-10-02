import json

from android_obfuscheck import cli
from android_obfuscheck.classinfo import ClassInfo
from android_obfuscheck.report import ROOT_PACKAGE, apply_gate, build_report, render_markdown

CLASSES = [
    ClassInfo("a", ("a",), ("b",)),
    ClassInfo("b", ("a",), ()),
    ClassInfo("com.example.app.MainActivity", ("onCreate",), ()),
    ClassInfo("com.example.app.c", ("a",), ()),
    ClassInfo("androidx.core.Foo", ("bar",), ()),
]
VERDICTS = [True, True, False, True, False]


def _report(**kw):
    return build_report("app.apk", "heuristic", CLASSES, VERDICTS, **kw)


def test_default_excludes_skip_libraries():
    r = _report()
    assert r.total_classes == 5
    assert r.scoped_classes == 4
    assert r.coverage == 0.75
    assert r.unobfuscated == ["com.example.app.MainActivity"]
    by_pkg = {p.package: p for p in r.packages}
    assert by_pkg[ROOT_PACKAGE].coverage == 1.0
    assert by_pkg["com.example.app"].coverage == 0.5


def test_include_prefix_is_segment_aware():
    assert _report(include=["com.example"]).scoped_classes == 2
    assert _report(include=["com.exam"]).scoped_classes == 0


def test_gate_min_coverage_and_baseline():
    assert apply_gate(_report(), min_coverage=0.7).passed
    assert not apply_gate(_report(), min_coverage=0.8).passed
    assert apply_gate(_report(), baseline={"coverage": 0.78}, max_drop=0.05).passed
    failed = apply_gate(_report(), baseline={"coverage": 0.9}, max_drop=0.05)
    assert "dropped" in failed.failures[0]


def test_gate_fails_on_empty_scope():
    assert not apply_gate(_report(include=["org.none"])).passed


def test_markdown_mentions_failures():
    md = render_markdown(apply_gate(_report(), min_coverage=0.9), list_unobfuscated=5)
    assert "❌" in md and "below the minimum" in md and "MainActivity" in md


def test_cli_scan(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("android_obfuscheck.extract.extract_classes", lambda _: CLASSES)
    out = tmp_path / "r.json"
    code = cli.main(["scan", "app.apk", "--min-coverage", "0.7", "--json-out", str(out)])
    assert code == cli.EXIT_OK
    assert json.loads(out.read_text())["coverage"] == 0.75
    assert "PASS" in capsys.readouterr().out

    code = cli.main(["scan", "app.apk", "--baseline", str(out), "--min-coverage", "0.9"])
    assert code == cli.EXIT_GATE_FAILED


def test_cli_scan_rejects_non_dex(tmp_path):
    bogus = tmp_path / "x.apk"
    bogus.write_bytes(b"not a zip")
    assert cli.main(["scan", str(bogus)]) == cli.EXIT_ERROR
