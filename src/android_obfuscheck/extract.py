"""Pull class/method/field names out of APK, AAB, JAR-of-dex or raw dex files."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

from android_obfuscheck.classinfo import ClassInfo

DEX_MAGIC = b"dex\n"


class ExtractionError(Exception):
    pass


def iter_dex_blobs(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(entry_name, bytes)`` for every dex in the archive (all of multidex, AAB modules)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for entry in sorted(zf.namelist()):
                if entry.endswith(".dex"):
                    yield entry, zf.read(entry)
        return
    data = path.read_bytes()
    if data[:4] != DEX_MAGIC:
        raise ExtractionError(f"{path} is neither a zip archive nor a dex file")
    yield path.name, data


def _quiet_androguard() -> None:
    try:
        from loguru import logger

        logger.disable("androguard")
    except ImportError:  # pragma: no cover - loguru ships with androguard 4
        pass


def extract_classes(path: str | Path) -> list[ClassInfo]:
    from androguard.core.dex import DEX

    _quiet_androguard()
    path = Path(path)
    classes: list[ClassInfo] = []
    found = False
    for _, blob in iter_dex_blobs(path):
        found = True
        dex = DEX(blob)
        for c in dex.get_classes():
            classes.append(
                ClassInfo.from_descriptor(
                    c.get_name(),
                    (m.get_name() for m in c.get_methods()),
                    (f.get_name() for f in c.get_fields()),
                )
            )
    if not found:
        raise ExtractionError(f"no .dex entries found in {path}")
    return classes
