"""Download audio via yt-dlp (search YouTube)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _yt_dlp_cmd() -> list[str]:
    """Return the command prefix for invoking yt-dlp.

    Prefers the ``yt-dlp`` binary on PATH.  Falls back to
    ``python -m yt_dlp`` so it works inside virtualenvs on Windows
    where script wrappers may not be on PATH.
    """
    if shutil.which("yt-dlp") is not None:
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def download_audio_search(query: str, out_dir: Path, filename_stem: str = "yt_audio") -> Path:
    """Download best search hit as WAV into ``out_dir``; return path to the ``.wav`` file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / filename_stem)
    cmd = [
        *_yt_dlp_cmd(),
        f"ytsearch1:{query}",
        "-x",
        "--audio-format",
        "wav",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-o",
        f"{template}.%(ext)s",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wav = Path(template + ".wav")
    if wav.is_file():
        return wav
    found = next(out_dir.glob(f"{filename_stem}*.wav"), None)
    if found is not None:
        return found
    raise FileNotFoundError(f"yt-dlp did not produce WAV under {out_dir}")
