"""Rich helpers for CLI branding and small presentation polish."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

LOGO = """  ____                        ____ _                _            
 |  _ \\ _ __ _   _ _ __ ___  / ___| |__   __ _ _ __| |_ ___ _ __ 
 | | | | '__| | | | '_ ` _ \\| |   | '_ \\ / _` | '__| __/ _ \\ '__|
 | |_| | |  | |_| | | | | | | |___| | | | (_| | |  | ||  __/ |   
 |____/|_|   \\__,_|_| |_| |_|\\____|_| |_|\\__,_|_|   \\__\\___|_|   

""".rstrip()


def print_branding(console: Console) -> None:
    """Render the DrumCharter logo for interactive terminal sessions."""
    if not console.is_terminal:
        return

    if console.width < 92:
        console.print(
            Panel.fit(
                Text("DrumCharter", style="bold cyan"),
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        return

    console.print(
        Panel.fit(
            Text(LOGO, style="bold cyan"),
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_generation_summary(
    console: Console,
    *,
    song_name: str,
    artist_name: str,
    backend: str,
    separate_drums: bool,
    quantize: str,
    output_dir: Path,
) -> None:
    """Show a compact summary before kicking off generation."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Song", song_name)
    table.add_row("Artist", artist_name)
    table.add_row("Backend", backend)
    table.add_row("Separation", "on" if separate_drums else "off")
    table.add_row("Quantize", quantize)
    table.add_row("Output", str(output_dir))
    console.print(
        Panel.fit(
            table,
            title="Generate Chart",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def print_generated_chart(
    console: Console,
    *,
    song_name: str,
    artist_name: str,
    folder: Path,
) -> None:
    """Show a clear success panel once chart generation finishes."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold green", justify="right")
    table.add_column()
    table.add_row("Song", song_name)
    table.add_row("Artist", artist_name)
    table.add_row("Folder", str(folder))
    console.print(
        Panel.fit(
            table,
            title="Chart Ready",
            border_style="green",
            box=box.ROUNDED,
        )
    )
