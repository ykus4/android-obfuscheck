import json
import stat

from android_obfuscheck import cli
from android_obfuscheck.apkid import gate_apkid, parse_apkid_json
from android_obfuscheck.classinfo import ClassInfo

SAMPLE = {
    "apkid_version": "3.0.0",
    "files": [
        {"filename": "app.apk!classes.dex", "matches": {"compiler": ["r8"]}},
        {
            "filename": "app.apk!classes2.dex",
            "matches": {"compiler": ["r8 (possible dexmerge)"], "anti_vm": ["Build.MODEL check"]},
        },
    ],
}


def test_parse_merges_files():
    r = parse_apkid_json(SAMPLE)
    assert r.version == "3.0.0"
    assert r.compilers == ["r8", "r8 (possible dexmerge)"]
    assert r.findings["anti_vm"] == ["Build.MODEL check"]


def test_parse_splits_multi_tag_keys():
    r = parse_apkid_json({"files": [{"matches": {"anti_vm, anti_debug": ["x"]}}]})
    assert r.findings == {"anti_debug": ["x"], "anti_vm": ["x"]}


def test_gate():
    r = parse_apkid_json(SAMPLE)
    assert gate_apkid(r, ["r8"], []) == []
    assert "expected one of dx" in gate_apkid(r, ["dx"], [])[0]
    assert gate_apkid(r, [], ["packer"]) == []
    assert "anti_vm" in gate_apkid(r, [], ["anti_vm"])[0]


def test_cli_with_fake_apkid(monkeypatch, tmp_path, capsys):
    fake = tmp_path / "apkid"
    payload = {"apkid_version": "x", "files": [{"matches": {"compiler": ["dexlib 2.x"]}}]}
    fake.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(
        "android_obfuscheck.extract.extract_classes", lambda _: [ClassInfo("a", ("a", "b"), ())]
    )

    code = cli.main(["scan", "app.apk", "--apkid-bin", str(fake), "--require-compiler", "r8"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_GATE_FAILED
    assert "dexlib 2.x" in out and "expected one of r8" in out


def test_cli_missing_apkid(monkeypatch):
    monkeypatch.setattr("android_obfuscheck.extract.extract_classes", lambda _: [ClassInfo("a")])
    assert cli.main(["scan", "app.apk", "--apkid", "--apkid-bin", "nope-xyz"]) == cli.EXIT_ERROR
