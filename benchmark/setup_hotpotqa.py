from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "pilot60_manifest.json"
TARGET = ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]

    expected = str(dataset["sha256"]).lower()
    urls = [
        str(dataset["source_url"]),
        str(dataset["archive_url"]),
    ]

    if TARGET.exists():
        actual = sha256_file(TARGET)
        if actual == expected:
            print(f"HotpotQA already present and verified: {actual}")
            return
        raise RuntimeError(
            f"Existing HotpotQA file has wrong SHA256: {actual}; expected {expected}"
        )

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".json.tmp")

    last_error: Exception | None = None

    for url in urls:
        try:
            print(f"Downloading HotpotQA from: {url}")
            urllib.request.urlretrieve(url, temporary)

            actual = sha256_file(temporary)
            if actual != expected:
                raise RuntimeError(
                    f"Downloaded HotpotQA SHA256 mismatch: {actual}; expected {expected}"
                )

            temporary.replace(TARGET)
            print(f"HotpotQA verified: {actual}")
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)

    raise RuntimeError("Unable to obtain verified HotpotQA dataset") from last_error


if __name__ == "__main__":
    main()
