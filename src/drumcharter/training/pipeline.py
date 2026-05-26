"""Data preparation pipeline for drum transcription training.

Discovers Rock Band song archives, extracts MIDI and audio files,
computes spectrograms and target label matrices, and caches the
results for subsequent training runs.
"""

from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    cache_dir: Path
    process_workers: int = 6
    force: bool = False


@dataclass
class PipelineResult:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _load_manifest(cache_dir):
    p = cache_dir / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_manifest(cache_dir, manifest):
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _process_archive(archive_path: str, cache_dir: str, force: bool) -> list[dict]:
    from pathlib import Path
    from drumcharter.training.discovery import discover_songs
    from drumcharter.training.extract import extract_archive
    from drumcharter.training.rb_midi import has_pro_markers
    from drumcharter.training.spectrogram import process_single_song, song_hash

    archive = Path(archive_path)
    results = []
    tmp = tempfile.TemporaryDirectory(prefix="drumcharter-worker-")
    try:
        extract_archive(archive, Path(tmp.name))
        songs = discover_songs(Path(tmp.name))
        for song in songs:
            song.source_archive = archive.name
            h = song_hash(song.path, stable_key=archive.name + "/" + song.song_name)
            entry = process_single_song(
                h=h, drum_stem_paths=[str(p) for p in song.drum_audio_paths],
                midi_path=str(song.midi_path), cache_dir=cache_dir,
                source_archive=archive.name, song_name=song.song_name,
                has_pro=has_pro_markers(song.midi_path), force=force,
            )
            if entry:
                results.append(entry)
    finally:
        tmp.cleanup()
    return results


def _process_song_direct(song_args: dict, cache_dir: str, force: bool) -> dict | None:
    from drumcharter.training.spectrogram import process_single_song
    return process_single_song(
        h=song_args["h"], drum_stem_paths=song_args["drum_stem_paths"],
        midi_path=song_args["midi_path"], cache_dir=cache_dir,
        source_archive=song_args["source_archive"], song_name=song_args["song_name"],
        has_pro=song_args["has_pro"], force=force,
    )


def run_pipeline(source_dirs: list[Path], config: PipelineConfig) -> PipelineResult:
    from drumcharter.training.discovery import discover_songs
    from drumcharter.training.extract import list_archives
    from drumcharter.training.rb_midi import has_pro_markers
    from drumcharter.training.spectrogram import song_hash

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(config.cache_dir)
    result = PipelineResult()

    extracted_args: list[dict] = []
    all_archives: list[Path] = []

    for source_dir in source_dirs:
        songs = discover_songs(source_dir)
        for song in songs:
            h = song_hash(song.path)
            if h in manifest and not config.force:
                result.skipped += 1
                continue
            extracted_args.append({
                "h": h, "drum_stem_paths": [str(p) for p in song.drum_audio_paths],
                "midi_path": str(song.midi_path), "source_archive": song.source_archive,
                "song_name": song.song_name, "has_pro": has_pro_markers(song.midi_path),
            })
        all_archives.extend(list_archives(source_dir))

    cached_archives: set[str] = set()
    if not config.force:
        for entry in manifest.values():
            src = entry.get("source_archive")
            if src:
                cached_archives.add(src)

    archives_to_process = [a for a in all_archives if a.name not in cached_archives]
    for a in all_archives:
        if a.name in cached_archives:
            result.skipped += 1

    total = len(extracted_args) + len(archives_to_process)
    if total == 0 and result.skipped == 0:
        return result

    cache_str = str(config.cache_dir)

    pbar = tqdm(
        total=total,
        desc="Songs",
        unit="song",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    with ProcessPoolExecutor(max_workers=config.process_workers) as executor:
        futures = {}
        for a in archives_to_process:
            futures[executor.submit(_process_archive, str(a), cache_str, config.force)] = a.name
        for args in extracted_args:
            futures[executor.submit(_process_song_direct, args, cache_str, config.force)] = args["song_name"]

        for future in as_completed(futures):
            name = futures[future]
            try:
                ret = future.result()
                if isinstance(ret, list):
                    for entry in ret:
                        h = entry.pop("hash")
                        manifest[h] = entry
                        result.processed += 1
                    pbar.set_postfix_str(f" {name[:30]}", refresh=False)
                    pbar.update(1)
                elif ret:
                    h = ret.pop("hash")
                    manifest[h] = ret
                    result.processed += 1
                    pbar.set_postfix_str(f" {ret.get('song_name', name)[:30]}", refresh=False)
                    pbar.update(1)
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{name}: {exc}")
                pbar.update(1)

        _save_manifest(config.cache_dir, manifest)

    pbar.close()
    return result
