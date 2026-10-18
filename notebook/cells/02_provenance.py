# ----------------------------------------------------------------------------------------------
# Load the raw file and verify it is the file we think it is.
# ----------------------------------------------------------------------------------------------
DATASET_META = {
    "name": "CDC Diabetes Health Indicators",
    "repository": "UCI Machine Learning Repository",
    "dataset_id": 891,
    "doi": "10.24432/C53919",
    "source_url": "https://archive.ics.uci.edu/static/public/891/data.csv",
    "underlying_survey": "CDC Behavioural Risk Factor Surveillance System (BRFSS) 2015",
    "license": "UCI terms of use / CDC public data",
}

def sha256_of(path: Path) -> str:
    """Streaming SHA-256 so a 13 MB file does not need to be held in memory twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
EXPECTED_ROWS = 253_680


if not DATA_RAW.exists():
    # Hosted runtimes start empty. Fetch from the UCI static host rather than failing, then verify.
    import ssl
    import urllib.request

    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Raw file absent -- downloading {DATASET_META['source_url']}")
    with urllib.request.urlopen(DATASET_META["source_url"],
                                context=ssl.create_default_context(), timeout=300) as response:
        DATA_RAW.write_bytes(response.read())
    print(f"Downloaded {DATA_RAW.stat().st_size/1e6:.2f} MB to {DATA_RAW}")

RAW_SHA256 = sha256_of(DATA_RAW)
assert RAW_SHA256 == EXPECTED_SHA256, (
    f"Digest mismatch: expected {EXPECTED_SHA256}, got {RAW_SHA256}. The file is not the dataset "
    f"this study used, so every number downstream would be wrong."
)

raw = pd.read_csv(DATA_RAW)
assert len(raw) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows, found {len(raw):,}"

print("File      :", DATA_RAW.name)
print("Size      :", f"{DATA_RAW.stat().st_size/1e6:.2f} MB")
print("SHA-256   :", RAW_SHA256)
print("Shape     :", f"{raw.shape[0]:,} rows x {raw.shape[1]} columns")
print()
print("Columns   :")
print(", ".join(raw.columns))
