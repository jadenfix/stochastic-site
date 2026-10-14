#!/usr/bin/env python3
"""Capture the deterministic offline workflow as truthful demo media."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo"
WIDTH, HEIGHT, FPS = 1280, 720, 24


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canvas(
    title: str, kicker: str, step: str
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07131f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (55, 42, 1225, 678), 28, fill="#0e2234", outline="#24475f", width=2
    )
    draw.text((92, 78), kicker.upper(), font=font(19, bold=True), fill="#57dbc4")
    draw.text((92, 116), title, font=font(42, bold=True), fill="#f4f8fb")
    draw.text((1082, 82), step, font=font(18, bold=True), fill="#9eb6c8")
    return image, draw


def add_scope(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle((92, 600, 1188, 642), 12, fill="#112f42")
    draw.text(
        (114, 611),
        "SEEDED FICTIONAL PRICES • PROVIDER-FREE • NO ORDERS • NO PERFORMANCE CLAIM",
        font=font(16, bold=True),
        fill="#9fe7d9",
    )


def fit_image(base: Image.Image, source: Path, box: tuple[int, int, int, int]) -> None:
    chart = Image.open(source).convert("RGB")
    chart.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    x = box[0] + (box[2] - box[0] - chart.width) // 2
    y = box[1] + (box[3] - box[1] - chart.height) // 2
    base.paste(chart, (x, y))


def render_cards(
    stdout: str,
    metrics: dict[str, float],
    provenance: dict[str, object],
    equity_chart: Path,
    states_chart: Path,
    guard_exit: int,
    before_hash: str,
    after_hash: str,
) -> list[Image.Image]:
    cards: list[Image.Image] = []

    image, draw = canvas(
        "Regime research, end to end", "Actual checked-in workflow", "01 / 05"
    )
    draw.text(
        (92, 205), "python -m src.cli demo", font=font(30, bold=True), fill="#f4f8fb"
    )
    draw.text((92, 270), "720 generated hourly bars", font=font(26), fill="#bed0dc")
    draw.text(
        (92, 315), "70% chronological training prefix", font=font(26), fill="#bed0dc"
    )
    draw.text(
        (92, 360),
        "held-out causal filtering + one-bar execution lag",
        font=font(26),
        fill="#bed0dc",
    )
    draw.text((92, 405), "embedded portable HTML report", font=font(26), fill="#bed0dc")
    add_scope(draw)
    cards.append(image)

    image, draw = canvas(
        "The real CLI completed", "Generated fixture receipt", "02 / 05"
    )
    lines = [line for line in stdout.splitlines() if line.strip()][-12:]
    y = 198
    for line in lines:
        display = line if len(line) <= 103 else line[:100] + "…"
        draw.text((92, y), display, font=font(18), fill="#c8d8e3")
        y += 31
    add_scope(draw)
    cards.append(image)

    image, draw = canvas(
        "Held-out equity path",
        "Seeded fictional fixture · actual report chart",
        "03 / 05",
    )
    draw.rounded_rectangle((92, 184, 1188, 564), 18, fill="#ffffff")
    fit_image(image, equity_chart, (112, 200, 1168, 548))
    metric_items = list(metrics.items())[:4]
    metric_labels = {
        "CAGR": "CAGR (decimal)",
        "Sharpe": "Sharpe",
        "MaxDrawdown": "Max drawdown",
        "HitRate": "Hit rate (decimal)",
    }
    x = 92
    for key, value in metric_items:
        draw.text(
            (x, 570),
            metric_labels.get(key, key.replace("_", " ").title()),
            font=font(13),
            fill="#91aabd",
        )
        draw.text((x, 591), f"{value:.3f}", font=font(20, bold=True), fill="#f4f8fb")
        x += 235
    cards.append(image)

    image, draw = canvas(
        "Forward-filtered regimes",
        "Seeded fictional fixture · actual report chart",
        "04 / 05",
    )
    draw.rounded_rectangle((92, 184, 1188, 564), 18, fill="#ffffff")
    fit_image(image, states_chart, (112, 200, 1168, 548))
    draw.text(
        (92, 584),
        f"{provenance['evaluation_rows']} held-out rows • {provenance['state_count']} states • labels from training prefix only",
        font=font(18, bold=True),
        fill="#c7d8e4",
    )
    cards.append(image)

    image, draw = canvas(
        "Source artifacts stay intact", "Repaired overwrite boundary", "05 / 05"
    )
    draw.text((92, 210), "Attempt", font=font(16, bold=True), fill="#8fa9bb")
    draw.text(
        (92, 242),
        "report --output <price parquet>",
        font=font(27, bold=True),
        fill="#f4f8fb",
    )
    draw.text(
        (92, 320),
        f"Rejected with exit {guard_exit}",
        font=font(30, bold=True),
        fill="#57dbc4",
    )
    draw.text((92, 382), "SHA-256 before", font=font(15), fill="#8fa9bb")
    draw.text((92, 407), before_hash, font=font(18), fill="#c9d9e4")
    draw.text((92, 457), "SHA-256 after", font=font(15), fill="#8fa9bb")
    draw.text((92, 482), after_hash, font=font(18), fill="#c9d9e4")
    draw.text(
        (92, 540),
        "Direct, symlink, and hard-link aliases are regression-tested.",
        font=font(20),
        fill="#bed0dc",
    )
    add_scope(draw)
    cards.append(image)
    return cards


def encode(cards: list[Image.Image], work: Path) -> None:
    frames = work / "frames"
    frames.mkdir()
    frames_per_card = 48
    transition = 8
    index = 0
    for card_index, card in enumerate(cards):
        previous = cards[max(0, card_index - 1)]
        for local in range(frames_per_card):
            if card_index and local < transition:
                frame = Image.blend(previous, card, (local + 1) / transition)
            else:
                frame = card.copy()
            draw = ImageDraw.Draw(frame)
            progress = (index + 1) / (len(cards) * frames_per_card)
            draw.rounded_rectangle(
                (0, HEIGHT - 7, WIDTH * progress, HEIGHT), 2, fill="#57dbc4"
            )
            frame.save(frames / f"frame-{index:04d}.png", optimize=True)
            index += 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUTPUT / "regime-research-demo.mp4"
    gif = OUTPUT / "regime-research-demo.gif"
    poster = OUTPUT / "regime-research-demo-poster.png"
    cards[2].save(poster, optimize=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames / "frame-%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif),
        ],
        check=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="regime-demo-") as temp:
        work = Path(temp)
        env = os.environ.copy()
        env.update(
            {
                "HMM_DATA_DIR": str(work / "artifacts"),
                "HMM_MODELS_DIR": str(work / "models"),
                "HMM_REPORTS_DIR": str(work / "reports"),
                "MPLCONFIGDIR": str(work / "mpl"),
                "MPLBACKEND": "Agg",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        demo = run([sys.executable, "-m", "src.cli", "demo"], env=env)
        if demo.returncode:
            raise SystemExit(demo.stdout + demo.stderr)

        price = work / "artifacts" / "binance_DEMO_1h.parquet"
        before = sha256(price)
        guard = run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "report",
                "--symbol",
                "DEMO",
                "--output",
                str(price),
            ],
            env=env,
        )
        after = sha256(price)
        if guard.returncode == 0 or before != after:
            raise SystemExit("Report overwrite guard did not reject without mutation")

        metrics = json.loads(
            (work / "artifacts" / "backtests" / "latest_metrics.json").read_text()
        )
        provenance = json.loads(
            (work / "artifacts" / "backtests" / "latest_provenance.json").read_text()
        )
        cards = render_cards(
            demo.stdout,
            metrics,
            provenance,
            work / "reports" / "equity_curve.png",
            work / "reports" / "states.png",
            guard.returncode,
            before,
            after,
        )
        encode(cards, work)
        print(
            json.dumps(
                {
                    "scope": "seeded fictional prices; provider-free; no orders",
                    "metrics": metrics,
                    "provenance": provenance,
                    "overwrite_guard_exit": guard.returncode,
                    "source_sha256_before": before,
                    "source_sha256_after": after,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
