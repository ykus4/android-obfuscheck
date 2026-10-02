"""The per-class record that every detector consumes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Compiler-generated method names carry no signal about obfuscation.
SPECIAL_METHODS = frozenset({"<init>", "<clinit>"})


@dataclass(frozen=True)
class ClassInfo:
    """A class as seen in dex: dotted name plus its method and field names."""

    name: str  # e.g. "com.example.Foo$Bar"
    methods: tuple[str, ...] = field(default_factory=tuple)
    fields: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_descriptor(cls, descriptor: str, methods, fields) -> ClassInfo:
        """Build from a dex type descriptor such as ``Lcom/example/Foo;``."""
        name = descriptor
        if name.startswith("L") and name.endswith(";"):
            name = name[1:-1]
        return cls(name.replace("/", "."), tuple(methods), tuple(fields))

    @property
    def package(self) -> str:
        return self.name.rpartition(".")[0]

    @property
    def simple_name(self) -> str:
        return self.name.rpartition(".")[2]

    @property
    def name_parts(self) -> list[str]:
        """Simple name split on ``$`` so inner/synthetic classes become separate tokens."""
        return [p for p in self.simple_name.split("$") if p]

    @property
    def member_names(self) -> list[str]:
        return [m for m in self.methods if m not in SPECIAL_METHODS] + list(self.fields)

    def to_text(self) -> str:
        """Flat token string used as model input."""
        return " ".join(self.name_parts + self.member_names)


# Format used by liansecurityOS/apk-obfucation-detection's new_train.csv:
#   "Class: Foo  Bar; Method: <init> a b Field: c d"
_UPSTREAM_ROW = re.compile(r"Class:\s*(.*?);?\s*Method:\s*(.*?)\s*Field:\s*(.*)$", re.S)


def parse_upstream_text(text: str) -> ClassInfo | None:
    m = _UPSTREAM_ROW.match(text.strip())
    if not m:
        return None
    parts = m.group(1).split()
    return ClassInfo("$".join(parts), tuple(m.group(2).split()), tuple(m.group(3).split()))
