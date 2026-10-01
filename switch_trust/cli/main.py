"""
CLI entry point for Switch Trust CLI.

Usage:
    python -m switch_trust.cli init
    python -m switch_trust.cli eval models list
    python -m switch_trust.cli eval run <model-evaluation-id>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

from dotenv import dotenv_values, find_dotenv
from rich.panel import Panel
from rich.text import Text

from switch_trust.cli import eval_cli, init_cli, scan_cli
from switch_trust.cli.console import CLI_WIDTH, console
from switch_trust.cli.utils import (
    ensure_client_id,
    ensure_telemetry_consent,
    get_client_id,
    get_switch_trust_dir,
    get_telemetry_consent,
    is_ci,
    legacy_env_vars_used,
)
from switch_trust.cli.version import VERSION
from switch_trust.eval.common.log import setup_file_logging, silence_noisy_loggers

if TYPE_CHECKING:
    from switch_trust.eval.db.json.repository_json import JsonRepository

logger = logging.getLogger(__name__)

_LOGO_LINES = [
    "███████╗██╗    ██╗██╗████████╗ ██████╗██╗  ██╗    ████████╗██████╗ ██╗   ██╗███████╗████████╗",
    "██╔════╝██║    ██║██║╚══██╔══╝██╔════╝██║  ██║    ╚══██╔══╝██╔══██╗██║   ██║██╔════╝╚══██╔══╝",
    "███████╗██║ █╗ ██║██║   ██║   ██║     ███████║       ██║   ██████╔╝██║   ██║███████╗   ██║",
    "╚════██║██║███╗██║██║   ██║   ██║     ██╔══██║       ██║   ██╔══██╗██║   ██║╚════██║   ██║",
    "███████║╚███╔███╔╝██║   ██║   ╚██████╗██║  ██║       ██║   ██║  ██║╚██████╔╝███████║   ██║",
    "╚══════╝ ╚══╝╚══╝ ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝",
]


_TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

_BUILTIN_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "builtin_config.json",
)


def _count(repo: JsonRepository) -> dict[str, int]:
    return {
        "models": len(repo.models.list()),
        "evaluations": len(repo.evaluations.list()),
        "detectors": len(repo.detectors.list()),
        "collections": len(repo.message_collections.list()),
        "assignments": len(repo.model_evaluations.list_all()),
    }


def _fmt_count(total: int, builtin: int, user: int) -> str:
    return f"{total} [dim]({builtin} builtin, {user} user)[/dim]"


def _print_logo() -> None:
    console.print()
    # Center the block by its widest line so every row shares one indent —
    # centering rows independently misaligns the art when they differ in width.
    logo_pad = max(0, (CLI_WIDTH - max(len(line) for line in _LOGO_LINES)) // 2)
    for line in _LOGO_LINES:
        console.print(" " * logo_pad + f"[bold cyan]{line}[/bold cyan]")
    version_text = f"v{VERSION}"
    pad = max(0, (CLI_WIDTH - len(version_text)) // 2)
    console.print(" " * pad + f"[dim][bold cyan]{version_text}[/bold cyan][/dim]")
    console.print()


def _print_config_info(
    merged: JsonRepository,
    builtin: JsonRepository,
    user: JsonRepository,
    config_path: str,
) -> None:
    m = _count(merged)
    b = _count(builtin)
    u = _count(user)

    console.print(f"[dim]Config:       {config_path}[/dim]")
    for label, key in [
        ("Models", "models"),
        ("Evaluations", "evaluations"),
        ("Detectors", "detectors"),
        ("Collections", "collections"),
        ("Assignments", "assignments"),
    ]:
        counts = _fmt_count(m[key], b[key], u[key])
        console.print(f"[dim]{label + ':':<14}{counts}[/dim]")
    console.print()


def _print_path(
    path: str,
) -> None:
    console.print(f"[dim]Path:         {path}[/dim]")
    console.print()


def _print_shutdown(
    elapsed: float,
    output_path: str | None,
    log_path: str | None,
) -> None:
    console.print()
    parts = [f"[dim]Completed in [bold cyan]{elapsed:.1f}s[/bold cyan][/dim]"]
    if output_path:
        parts.append(f"[dim]Results: [bold cyan]{output_path}[/bold cyan][/dim]")
    if log_path:
        parts.append(f"[dim]Logs: [bold cyan]{log_path}[/bold cyan][/dim]")
    console.print("  ".join(parts))
    console.print()


def _dispatch(args: argparse.Namespace) -> str | None:
    logger.info("Command: %s", " ".join(sys.argv))

    if args.command == "eval":
        return _dispatch_eval(args)
    elif args.command == "init":
        return _dispatch_init(args)
    elif args.command == "scan":
        return _dispatch_scan(args)
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        sys.exit(1)


def _dispatch_init(args: argparse.Namespace) -> str | None:
    init_cli.run_init()
    return None


def _dispatch_scan(args: argparse.Namespace) -> str | None:
    logger.info("switch-trust v%s | scan mode", VERSION)
    _print_path(args.path)
    return scan_cli.handle_scan(args)


def _dispatch_eval(args: argparse.Namespace) -> str | None:
    """Run the requested command. Returns output file path if any."""
    from switch_trust.eval.db.json.repository_json import (  # noqa: PLC0415 - deferred import cost
        JsonRepository,
    )

    if args.command != "eval":
        console.print(f"[red]Unknown command: {args.command}[/red]")
        sys.exit(1)

    config_path = getattr(args, "config", None) or str(
        init_cli.get_switch_trust_config_path(),
    )
    builtin = JsonRepository(_BUILTIN_CONFIG)
    user = JsonRepository(config_path)
    store = user.merge(builtin)

    _print_config_info(store, builtin, user, config_path)

    m = _count(store)
    logger.info(
        "switch-trust v%s | models=%d, evaluations=%d, assignments=%d",
        VERSION,
        m["models"],
        m["evaluations"],
        m["assignments"],
    )

    cmd = args.eval_cmd
    if cmd == "models":
        eval_cli.handle_models(args, store)
    elif cmd == "evaluations":
        eval_cli.handle_evaluations(args, store)
    elif cmd == "model-evaluations":
        eval_cli.handle_model_evaluations(
            args,
            store,
            user,
        )
    elif cmd == "run":
        return asyncio.run(eval_cli.handle_run(args, store))
    else:
        console.print(f"[red]Unknown eval subcommand: {cmd}[/red]")
        sys.exit(1)
    return None


def _print_error(error: BaseException) -> None:
    error_type = type(error).__name__
    error_msg = str(error) or "(no message)"

    body = Text()
    body.append(f"{error_type}: ", style="bold red")
    body.append(error_msg)

    panel = Panel(
        body,
        title="[bold red]Error[/bold red]",
        border_style="red",
        width=min(CLI_WIDTH, max(60, len(error_msg) + 20)),
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def _load_environment(override: bool = False) -> None:
    """Load env vars with precedence: real environment > local ``.env`` > global.

    Both dotfiles are read into a single mapping — a project-local ``.env`` from
    the current working directory layered over the global ``~/.switch-trust/.env`` so
    the local file wins for any shared key — and then applied to ``os.environ``
    in one pass. Applying the merged mapping ourselves (rather than two
    ``load_dotenv`` calls) is what lets a value already in ``os.environ`` — set
    from the shell, CI, or the command line (e.g. ``FOO=bar switch-trust ...``) —
    stay authoritative.

    The global file is resolved via ``get_switch_trust_dir()`` rather than
    ``get_switch_trust_env_path()`` so that a ``cwd/.env`` cannot mask the global
    file — its global-only keys (e.g. a persisted client id) must still load.

    ``override`` controls whether the merged dotfile values overwrite entries
    already in ``os.environ``. It stays ``False`` on normal startup so real
    shell/CI/command-line variables win over the dotfiles. It must be ``True``
    when reloading right after ``init`` (re)writes an env file — otherwise stale
    values loaded earlier this run would shadow the freshly written ones. Since
    ``init`` may write to *either* the local or the global file, the reload has
    to be able to pick up updates from both, which is why the flag applies to
    the merged result rather than to one file.
    """
    merged: dict[str, str | None] = {}

    global_env = get_switch_trust_dir() / ".env"
    if global_env.exists():
        merged.update(dotenv_values(global_env))

    local_dotenv = find_dotenv(usecwd=True)
    if local_dotenv:
        # Local layered last so it wins over the global file for shared keys.
        merged.update(dotenv_values(local_dotenv))

    for key, value in merged.items():
        if value is None:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="switch-trust",
        description="Switch Trust CLI — AI Agent Evaluation Framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    eval_cli.register(subparsers)
    init_cli.register(subparsers)
    scan_cli.register(subparsers)

    # Parse first: --version / --help / arg errors exit here, before we pay for
    # the heavy imports below (telemetry SDK, eval framework logging).
    args = parser.parse_args(argv)

    from switch_trust.cli.telemetry import (  # noqa: PLC0415 - deferred import cost
        command_span,
        emit_event,
        init_telemetry,
        record_error,
        record_interruption,
    )

    silence_noisy_loggers()

    ci = is_ci()
    switch_trust_env = init_cli.get_switch_trust_env_path()
    is_first_time = not switch_trust_env.exists()

    if is_first_time and args.command != "init":
        if ci:
            console.print(
                "[dim]CI environment detected — skipping interactive init.[/dim]",
            )
        else:
            console.print(
                "[yellow]First-time setup required. Running switch-trust init...[/yellow]",
            )
            console.print()
            init_cli.run_init()

    _load_environment()

    log_path = getattr(args, "log", None) or f"switch_trust_{_TIMESTAMP}.log"
    setup_file_logging(log_path)
    _print_logo()

    # On a first-time `switch-trust init` the .env doesn't exist yet and init (run
    # via _dispatch below) is the sole author of the client id.
    if not (is_first_time and args.command == "init"):
        ensure_client_id()
    if args.command != "init":
        ensure_telemetry_consent()

    if get_telemetry_consent():
        client_id = get_client_id()
        if client_id:
            init_telemetry(client_id, VERSION)

    # Separate any deprecated-env-var notes above from the command output below.
    if legacy_env_vars_used():
        console.print()

    def _try_init_telemetry_after_init() -> None:
        """Re-check consent after init creates the .env for the first time."""
        _load_environment(override=True)
        ensure_client_id()
        if get_telemetry_consent():
            client_id = get_client_id()
            if client_id:
                init_telemetry(client_id, VERSION)
                emit_event("init", is_ci=ci, is_first_time=is_first_time)

    subcommand_parts = []
    for attr in ("eval_cmd", "models_cmd", "evals_cmd", "me_cmd"):
        val = getattr(args, attr, None)
        if val:
            subcommand_parts.append(val)
    subcommand = " ".join(subcommand_parts) if subcommand_parts else None

    with command_span(
        args.command,
        subcommand,
        is_ci=ci,
        is_first_time=is_first_time,
    ) as span:
        t0 = time.monotonic()
        output_path: str | None = None
        try:
            output_path = _dispatch(args)
            if args.command == "init" and span is None:
                _try_init_telemetry_after_init()
        except KeyboardInterrupt:
            record_interruption(span)
            logger.info("Interrupted by user")
            console.print("\n[dim]Interrupted.[/dim]")
            sys.exit(130)
        except Exception as e:
            record_error(span, e)
            logger.critical(
                "Fatal error: %s: %s\n%s",
                type(e).__name__,
                e,
                traceback.format_exc(),
            )
            _print_error(e)
            elapsed = time.monotonic() - t0
            _print_shutdown(elapsed, output_path, log_path)
            sys.exit(1)
        elapsed = time.monotonic() - t0
        _print_shutdown(elapsed, output_path, log_path)
