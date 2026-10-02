import pytest

from android_obfuscheck.classinfo import ClassInfo, parse_upstream_text
from android_obfuscheck.detectors import HeuristicDetector, looks_obfuscated


@pytest.mark.parametrize("name", ["a", "Zb", "aa", "a1", "b12", "A0", "C5", "o90", "zze"])
def test_obfuscated_names(name):
    assert looks_obfuscated(name)


@pytest.mark.parametrize("name", ["onCreate", "MainActivity", "getValue", "url", "toString"])
def test_readable_names(name):
    assert not looks_obfuscated(name)


def test_from_descriptor():
    c = ClassInfo.from_descriptor("Lcom/example/Foo$Bar;", ["<init>", "run"], ["x"])
    assert c.name == "com.example.Foo$Bar"
    assert c.package == "com.example"
    assert c.name_parts == ["Foo", "Bar"]
    assert c.member_names == ["run", "x"]
    assert c.to_text() == "Foo Bar run x"


def test_parse_upstream_text():
    c = parse_upstream_text("Class: j b; Method: <init> a b c d Field: e")
    assert c.name_parts == ["j", "b"]
    assert c.methods == ("<init>", "a", "b", "c", "d")
    assert c.fields == ("e",)
    assert parse_upstream_text("garbage") is None


@pytest.mark.parametrize(
    "info, expected",
    [
        (ClassInfo("a.b.c", ("a", "b"), ("c",)), True),
        (ClassInfo("com.example.MainActivity", ("onCreate", "a", "b", "c"), ()), True),
        (ClassInfo("com.example.MainActivity", ("onCreate", "onResume"), ("binding",)), False),
        (ClassInfo("com.example.Point", ("<init>",), ("x",)), False),
    ],
)
def test_heuristic(info, expected):
    assert HeuristicDetector().is_obfuscated(info) is expected


def test_ml_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from android_obfuscheck.detectors import MLDetector
    from android_obfuscheck.train import train

    rows = ["text,label"]
    for i in range(40):
        rows.append(f'"Class: {chr(97 + i % 26)}; Method: <init> a b Field: c",1')
        rows.append(f'"Class: UserRepository{i}; Method: <init> findById save Field: dao",0')
    data = tmp_path / "train.csv"
    data.write_text("\n".join(rows))
    out = tmp_path / "m.joblib"

    meta = train(data, out)
    assert meta["rows"] == 80
    assert meta["evaluation"]["ml"]["accuracy"] >= 0.9

    det = MLDetector.load(out)
    assert det.predict(
        [ClassInfo("q", ("a", "b"), ("c",)), ClassInfo("com.x.OrderService", ("placeOrder",), ())]
    ) == [True, False]
