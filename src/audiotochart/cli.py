from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import click
import tempfile
from rich.console import Console
from rich.status import Status

from audiotochart.pipeline import STAGES, generate_drum_chart_folder
from audiotochart.download import download_audio_search
from audiotochart.inference.fake import FakeTranscriber

if TYPE_CHECKING:
    from audiotochart.inference.base import DrumTranscriber

console = Console()

BACKENDS: dict[str, type | None] = {
    "fake": FakeTranscriber,
}

try:
    from audiotochart.inference.adtof import AdtofTranscriber
    BACKENDS["adtof"] = AdtofTranscriber
except ImportError:
    BACKENDS["adtof"] = None

try:
    from audiotochart.inference.model import ModelTranscriber
    BACKENDS["model"] = ModelTranscriber
except ImportError:
    BACKENDS["model"] = None


def _resolve_backend(backend: str) -> type:
    cls = BACKENDS.get(backend)
    if cls is None:
        if backend == "adtof":
            console.print(
                "[red]ADTOF backend requires: [bold]uv sync --extra ai[/bold][/red]"
            )
        elif backend == "model":
            console.print(
                "[red]Model backend requires torch. Install the 'ai' extra: "
                "[bold]uv sync --extra ai[/bold][/red]"
            )
        else:
            available = ", ".join(BACKENDS)
            console.print(f"[red]Unknown backend '{backend}'. Available: {available}[/red]")
        raise SystemExit(1)
    return cls


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

def _run_generate(
    *,
    source_audio: Path,
    song_name: str,
    artist_name: str,
    dest_parent: Path,
    charter: str,
    bpm: float | None,
    from_midi: Path | None,
    backend: str = "fake",
    separate_drums: bool = False,
    device: str | None = None,
    keep_workdir: bool = False,
    model_dir: Path | None = None,
) -> Path:
    if backend == "model":
        if model_dir is None:
            console.print("[red]Model backend requires --model-dir[/red]")
            raise SystemExit(1)
        transcriber = ModelTranscriber(model_dir=model_dir, device=device)
    else:
        transcriber_cls = _resolve_backend(backend)
        transcriber = transcriber_cls()

    stage_labels = dict(STAGES)
    status: Status | None = None

    def _on_progress(stage: str, event: str) -> None:
        nonlocal status
        label = stage_labels.get(stage, stage)
        if event == "start":
            if status is None:
                status = console.status(f"[bold green] {label}...")
                status.__enter__()
            else:
                status.update(f"[bold green] {label}...")
        elif event == "done":
            console.print(f"[bold green]  {label}: done")

    try:
        return generate_drum_chart_folder(
            source_audio=source_audio,
            output_parent=dest_parent,
            song_name=song_name,
            artist_name=artist_name,
            charter=charter,
            bpm=bpm,
            from_midi=from_midi,
            transcriber=transcriber,
            separate_drums=separate_drums,
            device=device,
            keep_workdir=keep_workdir,
            on_progress=_on_progress,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1) from e
    finally:
        if status is not None:
            status.__exit__(None, None, None)

@click.group()
def cli() -> None:
    """Clone Hero Drum Chart Generator"""
    
@cli.command("generate")
@click.argument("audio", type=click.Path(path_type=Path, exists=False), default=None, required=False)
@click.option("--song", default=None, help="Song title for chart metadata")
@click.option("--artist", default=None, help="Artist name for chart metadata")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Parent folder for the new song directory")
@click.option("--bpm", type=float, default=None, help="BPM for chart timing (auto-detected if not provided)")
@click.option("--from-midi", type=click.Path(path_type=Path, exists=False), default=None, help="Developer path: build drum notes from a MIDI drum file")
@click.option("--backend", type=click.Choice(list(BACKENDS)), default="fake", help="Inference backend to use")
@click.option("--separate-drums/--no-separate-drums", default=False, help="Isolate drums with Demucs before transcription")
@click.option("--device", default=None, help="PyTorch device (cuda or cpu) for Demucs")
@click.option("--keep-workdir", is_flag=True, default=False, help="Preserve intermediate files for debugging")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None, help="Model directory for the 'model' backend")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed logging output")
def generate_cmd(
    audio: Path | None,
    song: str | None,
    artist: str | None,
    output: Path | None,
    bpm: float | None,
    from_midi: Path | None,
    backend: str,
    separate_drums: bool,
    device: str | None,
    keep_workdir: bool,
    model_dir: Path | None,
    verbose: bool,
) -> None:
    """Generate a first-pass drum chart from a local audio file."""
    _setup_logging(verbose)
    
    dest = output or Path.cwd()
    
    if audio is not None and not audio.is_file():
        console.print(f"[red]Not a file: {audio}[/red]")
        raise SystemExit(1)
    if from_midi is not None and not from_midi.is_file():
        console.print(f"[red]Not a file: {from_midi}[/red]")
        raise SystemExit(1)
    
    with ExitStack() as stack:
        if audio is None:
            if not song or not artist:
                console.print(
                    "[red]Provide a path to an audio file or pass both [bold]--song[/bold] and [bold]--artist[/bold] to search YouTube.[/red]"
                )
                raise SystemExit(1)
        
            # Search YouTube
            query = f"{artist} {song}"
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="audiotochart-yt-")))
            with console.status(f'[bold green] Download audio: "{query}"...'):
                wav_path = download_audio_search(query, tmp_path)
            audio = wav_path
            
            song_name = song
            artist_name = artist
            folder = _run_generate(
                source_audio=audio,
                song_name=song_name,
                artist_name=artist_name,
                dest_parent=dest,
                charter="AudioToChart (AI)",
                bpm=bpm,
                from_midi=from_midi,
                backend=backend,
                separate_drums=separate_drums,
                device=device,
                keep_workdir=keep_workdir,
                model_dir=model_dir,
            )
            console.print(f"[bold green]Generated chart[/bold green] -> {folder}")
            return

        assert audio is not None
        song_name = song or audio.stem
        artist_name = artist or "Unknown"
        
        folder = _run_generate(
            source_audio=audio,
            song_name=song_name,
            artist_name=artist_name,
            dest_parent=dest,
            charter="AudioToChart (AI)",
            bpm=bpm,
            from_midi=from_midi,
            backend=backend,
            separate_drums=separate_drums,
            device=device,
            keep_workdir=keep_workdir,
            model_dir=model_dir,
        )
        console.print(f"[bold green]Generated chart[/bold green] -> {folder}")


if __name__ == "__main__":
    cli()
