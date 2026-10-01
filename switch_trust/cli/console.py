from __future__ import annotations

import sys
import termios
import tty

from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme

CLI_WIDTH = 120

_theme = Theme(
    {
        "score.high": "bold green",
        "score.mid": "bold yellow",
        "score.low": "bold red",
        "status.finished": "bold green",
        "status.running": "bold blue",
        "status.error": "bold red",
        "status.waiting": "dim",
        "status.initializing": "bold cyan",
        "status.initialized": "cyan",
        "key": "dim",
    }
)

console = Console(theme=_theme, width=CLI_WIDTH)


def score_style(score: float) -> str:
    if score >= 0.7:
        return "score.high"
    if score >= 0.4:
        return "score.mid"
    return "score.low"


def status_style(status: str) -> str:
    normalized = status.lower().strip()
    return (
        f"status.{normalized}"
        if normalized
        in (
            "finished",
            "running",
            "error",
            "waiting",
            "initializing",
            "initialized",
        )
        else ""
    )


def severity_style(severity: str) -> str:
    s = severity.lower().strip()
    if s == "critical":
        return "bold red"
    if s == "high":
        return "red"
    if s == "medium":
        return "yellow"
    return "green"


def select(
    prompt: str,
    options: list[str],
    *,
    default_index: int = 0,
) -> str:
    if not sys.stdin.isatty():
        return Prompt.ask(prompt, choices=options, default=options[default_index])

    if len(options) == 1:
        return options[0]

    index = default_index
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _render() -> None:
        console.print(f"[bold cyan]{prompt}[/bold cyan]")
        for i, option in enumerate(options):
            if i == index:
                console.print(f"  [bold green]> {option}[/bold green]")
            else:
                console.print(f"    [dim]{option}[/dim]")

    def _clear_lines(n: int) -> None:
        for _ in range(n):
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.write("\r")
        sys.stdout.flush()

    try:
        _render()
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)

            if ch == "\r" or ch == "\n":
                break

            if ch == "\x03":
                raise KeyboardInterrupt

            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    index = (index - 1) % len(options)
                elif seq == "[B":
                    index = (index + 1) % len(options)

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _clear_lines(len(options) + 1)
                _render()
                tty.setraw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    _clear_lines(len(options) + 1)
    console.print(f"[bold cyan]{prompt}[/bold cyan]: [green]{options[index]}[/green]")

    return options[index]
