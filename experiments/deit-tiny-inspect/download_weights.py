#!/usr/bin/env python3
"""Fetch the official checkpoint named in facebookresearch/deit/models.py.

Downloaded source is evidence only: it is never imported/executed.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = "https://raw.githubusercontent.com/facebookresearch/deit/main/models.py"
WEIGHTS = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
        while chunk := response.read(1024*1024):
            target.write(chunk)
    temporary.replace(path)


def main():
    weights = ROOT / "weights"
    results = ROOT / "results"
    weights.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    source = results / "official-models-source.txt"
    checkpoint = weights / WEIGHTS.rsplit("/", 1)[1]
    fetch(SOURCE, source)
    assert WEIGHTS in source.read_text(), "Checkpoint URL not found in official source"
    fetch(WEIGHTS, checkpoint)
    checksum = sha256(checkpoint)
    assert checksum == "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97", "Checkpoint SHA256 mismatch"
    manifest = dict(model="deit_tiny_patch16_224", distilled=False,
                    source_url=SOURCE, source_sha256=sha256(source),
                    checkpoint_url=WEIGHTS, checkpoint_file=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=checksum,
                    checkpoint_bytes=checkpoint.stat().st_size,
                    quantized=False)
    (results / "model-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Official checkpoint ready: {checkpoint.name}, {checkpoint.stat().st_size} bytes")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
