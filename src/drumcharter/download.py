"""Download audio via yt-dlp (search YouTube)."""

from __future__ import annotations

import logging
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails to search or download audio."""


class DownloadSearchNoResults(DownloadError):
    """Raised when yt-dlp reports that a search query returned no hits."""


def _yt_dlp_cmd() -> list[str]:
    """Return the command prefix for invoking yt-dlp.

    Prefers the ``yt-dlp`` binary on PATH.  Falls back to
    ``python -m yt_dlp`` so it works inside virtualenvs on Windows
    where script wrappers may not be on PATH.
    """
    if shutil.which("yt-dlp") is not None:
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def _combined_output(stdout: str | None, stderr: str | None) -> str:
    parts = []
    if stdout:
        parts.append(stdout.strip())
    if stderr:
        parts.append(stderr.strip())
    return "\n".join(part for part in parts if part).strip()


def _looks_like_no_results(output: str) -> bool:
    lowered = output.lower()
    return (
        "no video results" in lowered
        or "no results found" in lowered
        or "did not match any videos" in lowered
        or "did not return any results" in lowered
    )


def download_audio_search(
    query: str,
    out_dir: Path,
    filename_stem: str = "yt_audio",
    *,
    verbose: bool = False,
) -> Path:
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
        "-o",
        f"{template}.%(ext)s",
    ]
    if verbose:
        cmd.append("--no-progress")
    else:
        cmd.extend(["--quiet", "--no-warnings"])

    logger.debug("Running downloader command: %s", shlex.join(cmd))
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    output = _combined_output(result.stdout, result.stderr)
    if verbose and output:
        logger.debug("yt-dlp output for query %r:\n%s", query, output)
    if result.returncode != 0:
        if _looks_like_no_results(output):
            raise DownloadSearchNoResults(f'No results found on YouTube for "{query}"')
        detail = output or f"yt-dlp exited with status {result.returncode}"
        raise DownloadError(f"yt-dlp failed for query {query!r}: {detail}")

    wav = Path(template + ".wav")
    if wav.is_file():
        return wav
    found = next(out_dir.glob(f"{filename_stem}*.wav"), None)
    if found is not None:
        return found
    if _looks_like_no_results(output):
        raise DownloadSearchNoResults(f'No results found on YouTube for "{query}"')
    detail = output or f"yt-dlp did not produce WAV under {out_dir}"
    raise DownloadError(detail)
