"""
Fetch the CDC Diabetes Health Indicators dataset from the UCI repository and verify it.

The raw CSV is deliberately **not** committed to the repository (see ``.gitignore``); it is
distributed through the Google Drive folder linked in the report.  This script is what makes
the pipeline reproducible from a clean checkout:

    python src/download_data.py

It writes ``data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`` and checks the SHA-256
digest against the value recorded in ``data/raw/SHA256SUMS``.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
EXPECTED_SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
EXPECTED_ROWS = 253_680

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        print(f"Already present: {DEST.name} ({DEST.stat().st_size/1e6:.2f} MB)")
    else:
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, DEST)
        print(f"Saved to {DEST} ({DEST.stat().st_size/1e6:.2f} MB)")

    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    n_rows = sum(1 for _ in open(DEST, "rb")) - 1

    print(f"SHA-256 : {digest}")
    print(f"Rows    : {n_rows:,}")

    ok = True
    if digest != EXPECTED_SHA256:
        print(f"MISMATCH: expected {EXPECTED_SHA256}")
        ok = False
    if n_rows != EXPECTED_ROWS:
        print(f"MISMATCH: expected {EXPECTED_ROWS:,} rows")
        ok = False

    (ROOT / "data" / "raw" / "SHA256SUMS").write_text(f"{digest}  {DEST.name}\n", encoding="utf-8")
    print("Verified." if ok else "Verification FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
