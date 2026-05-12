from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from audiotochart.pipeline import generate_drum_chart_folder

console = Console()

def _run_generate(
    *,
    source_audio: Path,
    song_name: str,
    artist_name: str,
    dest_parent: Path,
    charter: str,
    bpm: float,
) -> Path:

    try:
        return generate_drum_chart_folder(
            source_audio=source_audio,
            output_parent=dest_parent,
            song_name=song_name,
            artist_name=artist_name,
            charter=charter,
            bpm=bpm,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1) from e

@click.group()
def cli() -> None:
    """Clone Hero Drum Chart Generator"""
    
@cli.command("generate")
@click.argument("audio", type=click.Path(path_type=Path, exists=False), default=None, required=False)
@click.option("--song", default=None, help="Song title for chart metadata")
@click.option("--artist", default=None, help="Artist name for chart metadata")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Parent folder for the new song directory")
@click.option("--bpm", type=float, default=120.0, show_default=True, help="BPM for the generated fake chart")
def generate_cmd(
    audio: Path | None,
    song: str | None,
    artist: str | None,
    output: Path | None,
    bpm: float,
) -> None:
    """Generate a first-pass drum chart from a local audio file."""
    
    dest = output or Path.cwd()
    
    if audio is None:
        console.print("[red]Provide a path to a local audio file.[/red]")
        raise SystemExit(1)

    if not audio.is_file():
        console.print(f"[red]Not a file: {audio}[/red]")
        raise SystemExit(1)

    song_name = song or audio.stem
    artist_name = artist or "Unknown"
    folder = _run_generate(
        source_audio=audio,
        song_name=song_name,
        artist_name=artist_name,
        dest_parent=dest,
        charter="AudioToChart",
        bpm=bpm,
    )
    console.print(f"[bold green]Generated chart[/bold green] -> {folder}")


if __name__ == "__main__":
    cli()
