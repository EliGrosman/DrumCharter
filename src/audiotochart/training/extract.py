"""Archive extraction utilities for Rock Band song archives.

Discovers .7z archives and selectively extracts notes.mid, drum audio,
and song.ini files for downstream processing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def list_archives(root: Path) -> list[Path]:
    """Recursively list all .7z archives under a root directory.

    Args:
        root: The root directory to search.

    Returns:
        Sorted list of paths to .7z archives.

    Raises:
        NotADirectoryError: If root is not a valid directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    archives = sorted(root.rglob("*.7z"))
    log.info("Found %d .7z archives in %s", len(archives), root)
    return archives


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    selective: bool = True,
) -> Path:
    """Extract a .7z archive to a destination directory.

    In selective mode only notes.mid, drum audio, and song.ini files
    are extracted to minimise I/O.

    Args:
        archive_path: Path to the .7z archive.
        dest_dir: Destination directory for extracted files.
        selective: If True, extract only relevant game files.

    Returns:
        The destination directory path.

    Raises:
        RuntimeError: If 7z is not installed or extraction fails.
    """
    cmd = ["7z", "x", str(archive_path), f"-o{dest_dir}", "-y"]
    if selective:
        cmd.extend(["*/notes.mid", "*/drums*.ogg", "*/song.ini"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError(
            "7z (p7zip) is not installed. Install it with: sudo apt install p7zip-full"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"7z extraction failed for {archive_path.name}: {result.stderr[:500]}"
        )

    return dest_dir