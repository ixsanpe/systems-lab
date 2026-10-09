from pathlib import Path

import pytest
import pyzipper

from argmt.data import download as dl


def make_aes_zip(path: Path, members: dict[str, str], password: bytes) -> Path:
    with pyzipper.AESZipFile(path, "w", encryption=pyzipper.WZ_AES) as z:
        z.setpassword(password)
        for name, text in members.items():
            z.writestr(name, text)
    return path


def test_extract_member_decrypts_and_normalises(tmp_path):
    zp = make_aes_zip(tmp_path / "f.zip", {"dev/dev.arg_Latn": "Ola\r\nmundo"}, b"pw")
    out = dl.extract_member(zp, "dev/dev.arg_Latn", tmp_path / "out" / "dev.arg", b"pw")
    assert out.read_text() == "Ola\nmundo\n"


def test_extract_member_wrong_password(tmp_path):
    zp = make_aes_zip(tmp_path / "f.zip", {"a.txt": "x"}, b"pw")
    with pytest.raises(RuntimeError):
        dl.extract_member(zp, "a.txt", tmp_path / "a.txt", b"nope")


def test_alignment_check(tmp_path):
    for split in dl.FLORES_SPLITS:
        (tmp_path / f"{split}.spa_Latn").write_text("a\nb\n")
        (tmp_path / f"{split}.arg_Latn").write_text("a\nb\n")
    dl.check_flores_alignment(tmp_path)

    (tmp_path / "devtest.arg_Latn").write_text("a\n")
    with pytest.raises(RuntimeError, match="misaligned"):
        dl.check_flores_alignment(tmp_path)


def test_fetch_rejects_bad_checksum(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    src.write_bytes(b"tampered")
    monkeypatch.setattr(dl, "PILAR_RAW", src.parent.as_uri())
    archive = dl.Archive("src.zip", "dst.zip", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dl.fetch(archive, tmp_path / "dl")
    assert not (tmp_path / "dl" / "dst.zip").exists()
