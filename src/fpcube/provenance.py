"""Which code produced a result: git commit, whether the working tree was clean, and a content hash of the code.

The hash covers every tracked .py file under src/ and scripts/, so a result can be matched to a commit even when it
was produced before that commit was made (same hash = same code).
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def code_hash():
    h = hashlib.sha256()
    files = sorted(p for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py"))
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def info():
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"))
    return {"commit": _git("rev-parse", "--short", "HEAD"), "code_modified_since_commit": dirty,
            "code_sha256_16": code_hash()}
