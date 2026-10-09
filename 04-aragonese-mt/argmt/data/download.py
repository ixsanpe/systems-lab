"""Download the PILAR Aragonese corpora and the FLORES+ es/an dev+devtest sets.

Source: https://github.com/transducens/PILAR, pinned to a commit so the data is
reproducible. Every archive is checked against a SHA-256 before extraction.

Layout produced (all under data/raw/, git-ignored):

    pilar/crawled.arg_Latn        ~60k lines, language-ID-filtered web crawl
    pilar/literary.arg_Latn       ~25k lines, literary translations into Aragonese
    flores/{dev,devtest}.{spa_Latn,arg_Latn}
    flores/LICENSE_CC-BY-SA
    MANIFEST.json                 checksums + line counts

FLORES+ is shipped in an AES-encrypted zip (to keep it out of web crawls and
thus out of training sets). We decrypt it locally; never commit the plaintext.

Usage:
    uv run python -m argmt.data.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyzipper

from argmt import paths

PILAR_COMMIT = "1ee5bc1c678a2c5e45b94c4b15663e764e6f3205"  # 2025-05-28
PILAR_RAW = f"https://raw.githubusercontent.com/transducens/PILAR/{PILAR_COMMIT}"
FLORES_PASSWORD = b"multilingual machine translation"  # published in the PILAR README
FLORES_LANGS = ("spa_Latn", "arg_Latn")
FLORES_SPLITS = ("dev", "devtest")


@dataclass(frozen=True)
class Archive:
    url_path: str  # path in the PILAR repo, URL-encoded
    filename: str  # local name under data/raw/_downloads/
    sha256: str


ARCHIVES = {
    "crawled": Archive(
        "aragonese/crawled.zip",
        "pilar_aragonese_crawled.zip",
        "1fa1fabc94addb134c7b12f19ed882a0b29785fa58b3ae0a30ad81da35fb52df",
    ),
    "literary": Archive(
        "aragonese/literary.zip",
        "pilar_aragonese_literary.zip",
        "1446982dfcb25f9ce4375d8489a79c3934fde4af7f8eccea1783d9e9d77e0de3",
    ),
    "flores": Archive(
        "FLORES%2B.zip",
        "pilar_flores_plus.zip",
        "eb299e50d17bec41e6651f77ba9c1b193238253772bd628b1cbcd037e122b4d7",
    ),
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def fetch(archive: Archive, dest_dir: Path, force: bool = False) -> Path:
    """Download an archive (if missing or stale) and verify its checksum."""
    dest = dest_dir / archive.filename
    if dest.exists() and not force and sha256sum(dest) == archive.sha256:
        print(f"  cached   {dest.name}")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{PILAR_RAW}/{archive.url_path}"
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)

    actual = sha256sum(tmp)
    if actual != archive.sha256:
        tmp.unlink()
        raise RuntimeError(
            f"checksum mismatch for {archive.url_path}: expected {archive.sha256}, got {actual}"
        )
    tmp.replace(dest)
    return dest


def extract_member(zip_path: Path, member: str, dest: Path, password: bytes | None = None) -> Path:
    """Extract a single zip member to `dest`, normalising line endings to \\n."""
    with pyzipper.AESZipFile(zip_path) as z:
        if password:
            z.setpassword(password)
        text = z.read(member).decode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    dest.write_text(text, encoding="utf-8")
    return dest


def extract_all(downloads: dict[str, Path]) -> list[Path]:
    out = [
        extract_member(downloads["crawled"], "crawled.txt", paths.PILAR / "crawled.arg_Latn"),
        extract_member(downloads["literary"], "literary.txt", paths.PILAR / "literary.arg_Latn"),
        extract_member(
            downloads["flores"],
            "LICENSE_CC-BY-SA",
            paths.FLORES / "LICENSE_CC-BY-SA",
            FLORES_PASSWORD,
        ),
    ]
    for split in FLORES_SPLITS:
        for lang in FLORES_LANGS:
            out.append(
                extract_member(
                    downloads["flores"],
                    f"{split}/{split}.{lang}",
                    paths.FLORES / f"{split}.{lang}",
                    FLORES_PASSWORD,
                )
            )
    return out


def check_flores_alignment(flores_dir: Path) -> None:
    """Both sides of each split must have the same number of lines."""
    for split in FLORES_SPLITS:
        counts = {lang: count_lines(flores_dir / f"{split}.{lang}") for lang in FLORES_LANGS}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"FLORES+ {split} is misaligned: {counts}")


def write_manifest(files: list[Path], root: Path) -> Path:
    manifest = {
        "source": "https://github.com/transducens/PILAR",
        "commit": PILAR_COMMIT,
        "files": {
            str(p.relative_to(root)): {"sha256": sha256sum(p), "lines": count_lines(p)}
            for p in sorted(files)
        },
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    print(f"PILAR @ {PILAR_COMMIT[:12]}")
    downloads = {k: fetch(a, paths.DOWNLOADS, args.force) for k, a in ARCHIVES.items()}

    print("Extracting")
    files = extract_all(downloads)
    check_flores_alignment(paths.FLORES)
    manifest = write_manifest(files, paths.RAW)

    for name, meta in json.loads(manifest.read_text())["files"].items():
        print(f"  {meta['lines']:>7,}  {name}")
    print(f"Wrote {manifest.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
