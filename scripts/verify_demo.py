#!/usr/bin/env python3
"""Verify source pinning and fully decode the checked-in demo media."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, object]:
    output = command(
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    )
    return json.loads(output)


def main() -> None:
    receipt = json.loads((DEMO / "capture.json").read_text())
    source = receipt["captured_source_commit"]
    if (
        command("git", "rev-parse", f"{source}^{{tree}}")
        != receipt["captured_source_tree"]
    ):
        raise SystemExit("Captured source tree does not match its commit")

    manifest = json.loads((DEMO / "source-manifest.json").read_text())
    for path, expected_blob in manifest["runtime_input_blobs"].items():
        captured_blob = command("git", "rev-parse", f"{source}:{path}")
        current_blob = command("git", "hash-object", path)
        if captured_blob != expected_blob or current_blob != expected_blob:
            raise SystemExit(f"Runtime input drift: {path}")

    for name, expected in receipt["media"].items():
        path = DEMO / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise SystemExit(f"Media digest mismatch: {name}")
        actual = probe(path)
        stream = actual["streams"][0]
        for key in ("codec_name", "width", "height", "nb_read_frames"):
            if str(stream[key]) != str(expected[key]):
                raise SystemExit(f"Media probe mismatch: {name} {key}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            check=True,
        )

    print(
        f"Verified {len(receipt['media'])} media files and "
        f"{len(manifest['runtime_input_blobs'])} source inputs at {source}"
    )


if __name__ == "__main__":
    main()
