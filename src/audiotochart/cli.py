from __future__ import annotations

import logging
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.status import Status

from audiotochart.config import DEFAULT_CHARTER, config_exists, load_config, save_config
from audiotochart.device import VALID_TORCH_DEVICES
from audiotochart.download import download_audio_search
from audiotochart.inference.fake import FakeTranscriber
from audiotochart.pipeline import STAGES, generate_drum_chart_folder
from audiotochart.postprocess import QUANTIZE_CHOICES
from audiotochart.separation import SeparationError

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
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("audiotochart").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )


def _resolve_onset_decoder_dir(
    *,
    backend: str,
    cfg: dict,
    explicit_dir: Path | None,
    disabled: bool,
) -> Path | None:
    if backend != "model" or disabled:
        return None
    if explicit_dir is not None:
        return explicit_dir.expanduser().resolve()

    configured = cfg.get("onset_decoder_dir")
    if not configured:
        return None
    candidate = Path(configured).expanduser()
    if candidate.exists():
        return candidate.resolve()
    console.print(
        "[yellow]Skipping onset decoder: "
        f"{candidate} was not found; using baseline model output.[/yellow]"
    )
    return None


def _run_generate(
    *,
    source_audio: Path,
    song_name: str,
    artist_name: str,
    dest_parent: Path,
    charter: str,
    bpm: float | None,
    from_midi: Path | None,
    backend: str = "model",
    separate_drums: bool | None = None,
    device: str | None = "auto",
    keep_workdir: bool = False,
    model_dir: Path | None = None,
    onset_decoder_dir: Path | None = None,
    quantize_divisor: int | None = 16,
    tom_consistency: bool = False,
) -> Path:
    transcriber_cls = _resolve_backend(backend)

    if separate_drums is None:
        separate_drums = (backend == "model")

    if backend == "model":
        if model_dir is None:
            console.print("[red]Model backend requires --model-dir[/red]")
            raise SystemExit(1)
        transcriber = transcriber_cls(
            model_dir=model_dir,
            device=device,
            tom_consistency=tom_consistency,
            onset_decoder_dir=onset_decoder_dir,
        )
    else:
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
            quantize_divisor=quantize_divisor,
            on_progress=_on_progress,
        )
    except SeparationError as e:
        console.print(f"[red]{e}[/red]")
        console.print(
            "[yellow]Drum separation requires Demucs + torch. "
            "Install: [bold]uv sync --extra ai[/bold]. "
            "Skip separation: [bold]--no-separate-drums[/bold][/yellow]"
        )
        raise SystemExit(1) from e
    except (RuntimeError, ImportError) as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1) from e
    finally:
        if status is not None:
            status.__exit__(None, None, None)


def _run_interactive(cfg: dict) -> dict:
    load_saved = True
    if config_exists():
        load_saved = Confirm.ask("Load saved settings from last run?", default=True)

    if load_saved:
        console.print("\n[bold]Choose audio source:[/bold]")
        console.print("  [1] Local audio file")
        console.print("  [2] YouTube search (requires song name + artist)")
        source_choice = Prompt.ask("Select", choices=["1", "2"], default="1")

        if source_choice == "2":
            song_name = Prompt.ask("Song name")
            artist_name = Prompt.ask("Artist name")
            source_audio = None
        else:
            audio_path: Path | None = None
            while audio_path is None:
                raw = Prompt.ask("Audio file path")
                p = Path(raw).expanduser().resolve()
                if p.is_file():
                    audio_path = p
                else:
                    console.print(f"[red]Not a file: {p}[/red]")
            source_audio = audio_path
            song_name = Prompt.ask("Song name", default=audio_path.stem)
            artist_name = Prompt.ask("Artist name", default="Unknown")

        return {
            "source_audio": source_audio,
            "song_name": song_name,
            "artist_name": artist_name,
            "backend": cfg.get("backend", "model"),
            "model_dir": Path(cfg["model_dir"]).expanduser().resolve() if cfg.get("model_dir") else None,
            "onset_decoder_dir": Path(cfg["onset_decoder_dir"]).expanduser().resolve() if cfg.get("onset_decoder_dir") else None,
            "separate_drums": cfg.get("separate_drums", True),
            "device": cfg.get("device", "auto"),
            "quantize": cfg.get("quantize", "1/16"),
            "tom_consistency": cfg.get("tom_consistency", False),
            "output_dir": Path(cfg.get("output_dir", ".")).expanduser().resolve(),
            "_settings_changed": False,
        }

    console.print("\n[bold]Choose audio source:[/bold]")
    console.print("  [1] Local audio file")
    console.print("  [2] YouTube search (requires song name + artist)")
    source_choice = Prompt.ask("Select", choices=["1", "2"], default="1")

    if source_choice == "2":
        song_name = Prompt.ask("Song name")
        artist_name = Prompt.ask("Artist name")
        source_audio = None
    else:
        audio_path = None
        while audio_path is None:
            raw = Prompt.ask("Audio file path")
            p = Path(raw).expanduser().resolve()
            if p.is_file():
                audio_path = p
            else:
                console.print(f"[red]Not a file: {p}[/red]")
        source_audio = audio_path
        song_name = Prompt.ask("Song name", default=audio_path.stem)
        artist_name = Prompt.ask("Artist name", default="Unknown")

    backend = Prompt.ask("Backend", choices=list(BACKENDS), default=cfg.get("backend", "model"))

    model_dir: str | None = cfg.get("model_dir")
    onset_decoder_dir: str | None = cfg.get("onset_decoder_dir")
    if backend == "model":
        raw = Prompt.ask("Model directory", default=model_dir or "")
        model_dir = str(Path(raw).expanduser().resolve()) if raw else model_dir
        raw_decoder = Prompt.ask("Onset decoder directory", default=onset_decoder_dir or "")
        onset_decoder_dir = (
            str(Path(raw_decoder).expanduser().resolve())
            if raw_decoder
            else onset_decoder_dir
        )

    separate_drums = Confirm.ask("Separate drums with Demucs?", default=cfg.get("separate_drums", True))
    device = Prompt.ask("Device", choices=list(VALID_TORCH_DEVICES), default=cfg.get("device", "auto"))
    quantize = Prompt.ask(
        "Quantization",
        choices=list(QUANTIZE_CHOICES),
        default=cfg.get("quantize", "1/16"),
    )
    tom_consistency = Confirm.ask("Enable tom consistency?", default=cfg.get("tom_consistency", False))

    output_raw = Prompt.ask("Output directory", default=cfg.get("output_dir", "."))
    output_dir = Path(output_raw).expanduser().resolve()

    return {
        "source_audio": source_audio,
        "song_name": song_name,
        "artist_name": artist_name,
        "backend": backend,
        "model_dir": Path(model_dir).expanduser().resolve() if model_dir else None,
        "onset_decoder_dir": Path(onset_decoder_dir).expanduser().resolve() if onset_decoder_dir else None,
        "separate_drums": separate_drums,
        "device": device,
        "quantize": quantize,
        "tom_consistency": tom_consistency,
        "output_dir": output_dir,
        "_settings_changed": True,
    }


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
@click.option(
    "--backend",
    type=click.Choice(list(BACKENDS)),
    default="model",
    show_default=True,
    help="Inference backend (fake = development/test only)",
)
@click.option(
    "--separate-drums/--no-separate-drums",
    default=None,
    help="Isolate drums with Demucs before transcription (default: on for model backend, off otherwise)",
)
@click.option(
    "--device",
    type=click.Choice(VALID_TORCH_DEVICES),
    default="auto",
    show_default=True,
    help="PyTorch device for Demucs and model backend",
)
@click.option("--keep-workdir", is_flag=True, default=False, help="Preserve intermediate files for debugging")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Model directory for the 'model' backend (default: saved config or bundled model)",
)
@click.option(
    "--onset-decoder-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional chord onset decoder directory (default: saved config if it exists)",
)
@click.option(
    "--no-onset-decoder",
    is_flag=True,
    default=False,
    help="Force baseline model output even if an onset decoder is configured",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed logging output")
@click.option(
    "--quantize",
    type=click.Choice(list(QUANTIZE_CHOICES), case_sensitive=False),
    default="1/16",
    show_default=True,
    help="Quantization grid subdivision (none = no snap)",
)
@click.option(
    "--tom-consistency/--no-tom-consistency",
    default=False,
    show_default=True,
    help="Enable/disable tom consistency post-processing",
)
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
    onset_decoder_dir: Path | None,
    no_onset_decoder: bool,
    verbose: bool,
    quantize: str,
    tom_consistency: bool,
) -> None:
    """Generate a first-pass drum chart from a local audio file.

    Run without arguments for an interactive walkthrough.
    """
    _setup_logging(verbose)

    user_config = load_config()

    if audio is None and song is None and artist is None:
        params = _run_interactive(user_config)
        interactive_cfg = dict(user_config)
        if params.get("onset_decoder_dir") is not None:
            interactive_cfg["onset_decoder_dir"] = str(params["onset_decoder_dir"])
        resolved_onset_decoder_dir = _resolve_onset_decoder_dir(
            backend=params["backend"],
            cfg=interactive_cfg,
            explicit_dir=onset_decoder_dir,
            disabled=no_onset_decoder,
        )

        with ExitStack() as stack:
            if params["source_audio"] is None:
                query = f"{params['artist_name']} {params['song_name']}"
                tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="audiotochart-yt-")))
                with console.status(f'[bold green] Download audio: "{query}"...'):
                    try:
                        wav_path = download_audio_search(query, tmp_path)
                    except FileNotFoundError:
                        console.print(f'[red]No results found on YouTube for "{query}"[/red]')
                        raise SystemExit(1)
                audio_file = wav_path
            else:
                audio_file = params["source_audio"]

            folder = _run_generate(
                source_audio=audio_file,
                song_name=params["song_name"],
                artist_name=params["artist_name"],
                dest_parent=params["output_dir"],
                charter=user_config.get("charter", DEFAULT_CHARTER),
                bpm=bpm,
                from_midi=from_midi,
                backend=params["backend"],
                separate_drums=params["separate_drums"],
                device=params["device"],
                keep_workdir=keep_workdir,
                model_dir=params["model_dir"],
                onset_decoder_dir=resolved_onset_decoder_dir,
                quantize_divisor=QUANTIZE_CHOICES[params["quantize"]],
                tom_consistency=params["tom_consistency"],
            )

        if params.get("_settings_changed", False):
            if Confirm.ask("Save these settings for next time?", default=True):
                save_config({
                    "backend": params["backend"],
                    "model_dir": str(params["model_dir"]) if params["model_dir"] else "",
                    "onset_decoder_dir": str(params["onset_decoder_dir"]) if params["onset_decoder_dir"] else "",
                    "device": params["device"],
                    "separate_drums": params["separate_drums"],
                    "quantize": params["quantize"],
                    "tom_consistency": params["tom_consistency"],
                    "charter": user_config.get("charter", DEFAULT_CHARTER),
                    "output_dir": str(params["output_dir"]),
                })

        console.print(f"[bold green]Generated chart[/bold green] -> {folder}")
        return

    dest = output or Path.cwd()

    if audio is not None and not audio.is_file():
        console.print(f"[red]Not a file: {audio}[/red]")
        raise SystemExit(1)
    if from_midi is not None and not from_midi.is_file():
        console.print(f"[red]Not a file: {from_midi}[/red]")
        raise SystemExit(1)

    if model_dir is None and backend == "model":
        cfg_dir = user_config.get("model_dir")
        if cfg_dir:
            model_dir = Path(cfg_dir)
    if separate_drums is None:
        separate_drums = user_config.get("separate_drums") if backend == "model" else (backend == "model")
    resolved_onset_decoder_dir = _resolve_onset_decoder_dir(
        backend=backend,
        cfg=user_config,
        explicit_dir=onset_decoder_dir,
        disabled=no_onset_decoder,
    )

    with ExitStack() as stack:
        if audio is None:
            if not song or not artist:
                console.print(
                    "[red]Provide a path to an audio file or pass both [bold]--song[/bold] and [bold]--artist[/bold] to search YouTube.[/red]"
                )
                raise SystemExit(1)

            query = f"{artist} {song}"
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="audiotochart-yt-")))
            with console.status(f'[bold green] Download audio: "{query}"...'):
                try:
                    wav_path = download_audio_search(query, tmp_path)
                except FileNotFoundError:
                    console.print(f'[red]No results found on YouTube for "{query}"[/red]')
                    raise SystemExit(1)
            audio = wav_path

            song_name = song
            artist_name = artist
            folder = _run_generate(
                source_audio=audio,
                song_name=song_name,
                artist_name=artist_name,
                dest_parent=dest,
                charter=DEFAULT_CHARTER,
                bpm=bpm,
                from_midi=from_midi,
                backend=backend,
                separate_drums=separate_drums,
                device=device,
                keep_workdir=keep_workdir,
                model_dir=model_dir,
                onset_decoder_dir=resolved_onset_decoder_dir,
                quantize_divisor=QUANTIZE_CHOICES[quantize],
                tom_consistency=tom_consistency,
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
            charter=DEFAULT_CHARTER,
            bpm=bpm,
            from_midi=from_midi,
            backend=backend,
            separate_drums=separate_drums,
            device=device,
            keep_workdir=keep_workdir,
            model_dir=model_dir,
            onset_decoder_dir=resolved_onset_decoder_dir,
            quantize_divisor=QUANTIZE_CHOICES[quantize],
            tom_consistency=tom_consistency,
        )
        console.print(f"[bold green]Generated chart[/bold green] -> {folder}")


if __name__ == "__main__":
    cli()
