"""
Subcommands for `switch-trust eval`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from switch_trust.cli.console import CLI_WIDTH, console
from switch_trust.cli.utils import get_switch_trust_config_path
from switch_trust.cli import llm_credentials
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.cli.http_status_kinds import StatusKind, status_kind
from switch_trust.eval.common.utils import resolve_env
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.models.model_errors import HOST_NOT_FOUND, ErrorKind
from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation
from switch_trust.eval.db.base.models.model_types import ModelType

if TYPE_CHECKING:
    # Annotations only — `handle_run` imports the runner lazily at call time,
    # and every annotation here is a string under `from __future__ import
    # annotations`, so this costs nothing at runtime.
    from switch_trust.cli.runner import CliRunResult
    from switch_trust.eval.core.models.model import ModelErrorAccumulator
    from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation
    from switch_trust.eval.db.json.repository_json import JsonRepository

_DEFAULT_CONFIG = str(get_switch_trust_config_path())

_DEFAULT_LOG = f"switch_trust_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"

_config_parent = argparse.ArgumentParser(add_help=False)
_config_parent.add_argument(
    "--config",
    default=_DEFAULT_CONFIG,
    help=f"Path to JSON config file (default: {_DEFAULT_CONFIG})",
)
_config_parent.add_argument(
    "--log",
    help=f"Log file path (default: {_DEFAULT_LOG})",
)


_tag_parent = argparse.ArgumentParser(add_help=False)
_tag_parent.add_argument(
    "--tag",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Filter by tag (repeatable)",
)


def _parse_tags(raw: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            # CliExit, not sys.exit: SystemExit is a BaseException, so it flies
            # past main's `except Exception` and the run is recorded to
            # telemetry as a success. A malformed KEY=VALUE is a bad
            # invocation, which is what argparse already exits 2 for.
            raise CliExit(
                ExitCode.USAGE,
                f"Invalid tag format: {item!r} (expected KEY=VALUE)",
            )
        k, v = item.split("=", 1)
        tags[k] = v
    return tags


def _matches_tags(
    entity_tags: dict[str, str],
    filter_tags: dict[str, str],
) -> bool:
    return all(entity_tags.get(k) == v for k, v in filter_tags.items())


def register(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluation commands",
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_cmd")
    eval_sub.required = True

    _register_models(eval_sub)
    _register_evaluations(eval_sub)
    _register_model_evaluations(eval_sub)
    _register_run(eval_sub)


# -- models -------------------------------------------------------


def _register_models(
    subparsers: argparse._SubParsersAction,
) -> None:
    models_parser = subparsers.add_parser(
        "models",
        help="Model commands",
    )
    models_sub = models_parser.add_subparsers(
        dest="models_cmd",
    )
    models_sub.required = True

    models_sub.add_parser(
        "list",
        help="List all models",
        parents=[_config_parent, _tag_parent],
    )

    show_parser = models_sub.add_parser(
        "show",
        help="Show model details",
        parents=[_config_parent],
    )
    show_parser.add_argument("id", help="Model ID")


def handle_models(
    args: argparse.Namespace,
    store: JsonRepository,
) -> None:
    if args.models_cmd == "list":
        _models_list(store, _parse_tags(args.tag))
    elif args.models_cmd == "show":
        _models_show(store, args.id)


def _models_list(
    store: JsonRepository,
    filter_tags: dict[str, str] | None = None,
) -> None:
    models = store.models.list()
    if filter_tags:
        models = [m for m in models if _matches_tags(m.tags, filter_tags)]
    if not models:
        console.print("[dim]No models configured.[/dim]")
        return
    table = Table(title="Models", show_lines=False, width=CLI_WIDTH)
    table.add_column("ID", style="dim", ratio=2, overflow="ellipsis", no_wrap=True)
    table.add_column("Name", style="bold", ratio=3, overflow="ellipsis", no_wrap=True)
    table.add_column("Type", ratio=1, overflow="ellipsis", no_wrap=True)
    for m in models:
        table.add_row(m.id, m.name, m.type.value)
    console.print(table)


def _models_show(store: JsonRepository, model_id: str) -> None:
    model_id = _resolve_id(
        model_id,
        [m.id for m in store.models.list()],
    )
    model = store.models.get(model_id)

    detail = Table.grid(padding=(0, 2))
    detail.add_column(style="key", justify="right")
    detail.add_column()
    detail.add_row("ID", model.id)
    detail.add_row("Name", model.name)
    detail.add_row("Type", model.type.value)
    detail.add_row("Model name", model.model_name)
    if model.host:
        detail.add_row("Host", model.host)
    if model.endpoint:
        detail.add_row("Endpoint", model.endpoint)
    if model.tags:
        detail.add_row("Tags", _fmt_tags(model.tags))
    if model.description:
        detail.add_row("Description", model.description)

    console.print(Panel(detail, title=f"[bold]{model.name}[/bold]", expand=False))

    me_view = store.model_evaluations.list_by_model(
        model.id,
        limit=100,
    )
    if me_view.items:
        table = Table(
            title=f"Connected evaluations ({me_view.total})",
            width=CLI_WIDTH,
        )
        table.add_column("ID", style="dim", ratio=2, overflow="ellipsis", no_wrap=True)
        table.add_column(
            "Name", style="bold", ratio=3, overflow="ellipsis", no_wrap=True
        )
        table.add_column("Evaluation", ratio=3, overflow="ellipsis", no_wrap=True)
        for item in me_view.items:
            eval_name = item.evaluation_ref.name if item.evaluation_ref else "?"
            eval_cell = Text(eval_name)
            eval_cell.append("\n")
            eval_cell.append(item.config.evaluation_id, style="dim")
            table.add_row(item.config.id, item.config.name, eval_cell)
        console.print(table)


# -- evaluations ---------------------------------------------------


def _register_evaluations(
    subparsers: argparse._SubParsersAction,
) -> None:
    evals_parser = subparsers.add_parser(
        "evaluations",
        help="Evaluation commands",
    )
    evals_sub = evals_parser.add_subparsers(
        dest="evals_cmd",
    )
    evals_sub.required = True

    evals_sub.add_parser(
        "list",
        help="List all evaluations",
        parents=[_config_parent, _tag_parent],
    )

    show_parser = evals_sub.add_parser(
        "show",
        help="Show evaluation details",
        parents=[_config_parent],
    )
    show_parser.add_argument("id", help="Evaluation ID")


def handle_evaluations(
    args: argparse.Namespace,
    store: JsonRepository,
) -> None:
    if args.evals_cmd == "list":
        _evaluations_list(store, _parse_tags(args.tag))
    elif args.evals_cmd == "show":
        _evaluations_show(store, args.id)


def _evaluations_list(
    store: JsonRepository,
    filter_tags: dict[str, str] | None = None,
) -> None:
    evaluations = store.evaluations.list()
    if filter_tags:
        evaluations = [e for e in evaluations if _matches_tags(e.tags, filter_tags)]
    if not evaluations:
        console.print("[dim]No evaluations configured.[/dim]")
        return
    table = Table(title="Evaluations", show_lines=False, width=CLI_WIDTH)
    table.add_column("ID", style="dim", ratio=2)
    table.add_column("Name", style="bold", ratio=3, overflow="ellipsis", no_wrap=True)
    table.add_column("Type", ratio=2, overflow="ellipsis", no_wrap=True)
    for e in evaluations:
        table.add_row(e.id, e.name, e.type.value)
    console.print(table)


def _evaluations_show(
    store: JsonRepository,
    eval_id: str,
) -> None:
    eval_id = _resolve_id(
        eval_id,
        [e.id for e in store.evaluations.list()],
    )
    evaluation = store.evaluations.get(eval_id)

    detail = Table.grid(padding=(0, 2))
    detail.add_column(style="key", justify="right")
    detail.add_column()
    detail.add_row("ID", evaluation.id)
    detail.add_row("Name", evaluation.name)
    detail.add_row("Type", evaluation.type.value)
    detail.add_row("Approach", evaluation.approach.value)
    if evaluation.tags:
        detail.add_row("Tags", _fmt_tags(evaluation.tags))
    if evaluation.description:
        detail.add_row("Description", evaluation.description)
    if evaluation.adversarial_goals:
        prompts = evaluation.num_prompts or 5
        turns = evaluation.max_turns or 5
        detail.add_row("Prompts", str(prompts))
        detail.add_row("Max turns", str(turns))

    console.print(Panel(detail, title=f"[bold]{evaluation.name}[/bold]", expand=False))

    me_view = store.model_evaluations.list_by_evaluation(
        evaluation.id,
        limit=100,
    )
    if me_view.items:
        table = Table(
            title=f"Connected models ({me_view.total})",
            width=CLI_WIDTH,
        )
        table.add_column("ID", style="dim", ratio=2, overflow="ellipsis", no_wrap=True)
        table.add_column(
            "Name", style="bold", ratio=3, overflow="ellipsis", no_wrap=True
        )
        table.add_column("Model", ratio=3, overflow="ellipsis", no_wrap=True)
        for item in me_view.items:
            model_name = item.model_ref.name if item.model_ref else "?"
            model_cell = Text(model_name)
            model_cell.append("\n")
            model_cell.append(item.config.model_id, style="dim")
            table.add_row(item.config.id, item.config.name, model_cell)
        console.print(table)


# -- model-evaluations --------------------------------------------

_model_selector_parent = argparse.ArgumentParser(
    add_help=False,
)
_model_selector_parent.add_argument(
    "--model",
    action="append",
    default=[],
    metavar="ID",
    help="Model ID (repeatable)",
)
_model_selector_parent.add_argument(
    "--model-tag",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Match models by tag (repeatable)",
)

_eval_selector_parent = argparse.ArgumentParser(
    add_help=False,
)
_eval_selector_parent.add_argument(
    "--eval",
    action="append",
    default=[],
    metavar="ID",
    help="Evaluation ID (repeatable)",
)
_eval_selector_parent.add_argument(
    "--eval-tag",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Match evaluations by tag (repeatable)",
)


def _register_model_evaluations(
    subparsers: argparse._SubParsersAction,
) -> None:
    me_parser = subparsers.add_parser(
        "model-evaluations",
        help="Model-evaluation assignment commands",
    )
    me_sub = me_parser.add_subparsers(dest="me_cmd")
    me_sub.required = True
    me_sub.add_parser(
        "list",
        help="List all model-evaluation assignments",
        parents=[
            _config_parent,
            _tag_parent,
            _model_selector_parent,
            _eval_selector_parent,
        ],
    )
    me_sub.add_parser(
        "attach",
        help="Assign evaluations to models",
        parents=[
            _config_parent,
            _model_selector_parent,
            _eval_selector_parent,
        ],
    )
    me_sub.add_parser(
        "detach",
        help="Remove evaluation assignments from models",
        parents=[
            _config_parent,
            _model_selector_parent,
            _eval_selector_parent,
        ],
    )


def handle_model_evaluations(
    args: argparse.Namespace,
    store: JsonRepository,
    user_store: JsonRepository,
) -> None:
    if args.me_cmd == "list":
        _model_evaluations_list(
            store,
            filter_tags=_parse_tags(args.tag),
            models=_resolve_models(args, store),
            evaluations=_resolve_evaluations(args, store),
        )
    elif args.me_cmd == "attach":
        _model_evaluations_attach(args, store, user_store)
    elif args.me_cmd == "detach":
        _model_evaluations_detach(args, store, user_store)


def _model_evaluations_list(
    store: JsonRepository,
    filter_tags: dict[str, str] | None = None,
    models: list | None = None,
    evaluations: list | None = None,
) -> None:
    items = store.model_evaluations.list_all()
    if filter_tags:
        items = [me for me in items if _matches_tags(me.tags, filter_tags)]
    if models:
        model_ids = {m.id for m in models}
        items = [me for me in items if me.model_id in model_ids]
    if evaluations:
        eval_ids = {e.id for e in evaluations}
        items = [me for me in items if me.evaluation_id in eval_ids]
    if not items:
        console.print("[dim]No model-evaluation assignments configured.[/dim]")
        return

    table = Table(
        title="Model-Evaluation Assignments",
        show_lines=False,
        width=CLI_WIDTH,
    )
    table.add_column("ID", style="dim", ratio=2, overflow="ellipsis", no_wrap=True)
    table.add_column("Name", style="bold", ratio=2, overflow="ellipsis", no_wrap=True)
    table.add_column("Model", ratio=2, overflow="ellipsis", no_wrap=True)
    table.add_column("Evaluation", ratio=3, overflow="ellipsis", no_wrap=True)

    for me in items:
        model_name = "?"
        eval_name = "?"
        try:
            model_name = store.models.get(
                me.model_id,
            ).name
        except KeyError:
            pass
        try:
            eval_name = store.evaluations.get(
                me.evaluation_id,
            ).name
        except KeyError:
            pass

        model_cell = Text(model_name)
        model_cell.append("\n")
        model_cell.append(me.model_id, style="dim")

        eval_cell = Text(eval_name)
        eval_cell.append("\n")
        eval_cell.append(me.evaluation_id, style="dim")

        table.add_row(me.id, me.name, model_cell, eval_cell)

    console.print(table)


def _resolve_models(
    args: argparse.Namespace,
    store: JsonRepository,
) -> list:
    model_ids = args.model
    model_tags = _parse_tags(args.model_tag)

    if not model_ids and not model_tags:
        return []

    models = []
    seen_ids: set[str] = set()

    for mid in model_ids:
        resolved = _resolve_id(
            mid,
            [m.id for m in store.models.list()],
        )
        try:
            m = store.models.get(resolved)
            if m.id not in seen_ids:
                models.append(m)
                seen_ids.add(m.id)
        except KeyError:
            raise ValueError(f"model {mid!r} not found") from None

    if model_tags:
        for m in store.models.list():
            if m.id not in seen_ids and _matches_tags(m.tags, model_tags):
                models.append(m)
                seen_ids.add(m.id)

    return models


def _resolve_evaluations(
    args: argparse.Namespace,
    store: JsonRepository,
) -> list:
    eval_ids = getattr(args, "eval", [])
    eval_tags = _parse_tags(args.eval_tag)

    if not eval_ids and not eval_tags:
        return []

    evaluations = []
    seen_ids: set[str] = set()

    for eid in eval_ids:
        resolved = _resolve_id(
            eid,
            [e.id for e in store.evaluations.list()],
        )
        try:
            e = store.evaluations.get(resolved)
            if e.id not in seen_ids:
                evaluations.append(e)
                seen_ids.add(e.id)
        except KeyError:
            raise ValueError(f"evaluation {eid!r} not found") from None

    if eval_tags:
        for e in store.evaluations.list():
            if e.id not in seen_ids and _matches_tags(e.tags, eval_tags):
                evaluations.append(e)
                seen_ids.add(e.id)

    return evaluations


def _model_evaluations_attach(
    args: argparse.Namespace,
    store: JsonRepository,
    user_store: JsonRepository,
) -> None:

    models = _resolve_models(args, store)
    evaluations = _resolve_evaluations(args, store)

    if not models:
        raise ValueError("no models matched — specify --model or --model-tag")
    if not evaluations:
        raise ValueError("no evaluations matched — specify --eval or --eval-tag")

    added: list[DbModelEvaluation] = []
    skipped = 0

    for m in models:
        for e in evaluations:
            me = DbModelEvaluation(
                model_id=m.id,
                evaluation_id=e.id,
                name=f"{m.name} / {e.name}",
            )
            if user_store.model_evaluations.add(me):
                added.append(me)
            else:
                skipped += 1

    if added:
        user_store.save()
        table = Table(
            title=f"Attached ({len(added)} new)",
            show_lines=False,
            width=CLI_WIDTH,
        )
        table.add_column("Model", ratio=2)
        table.add_column("Evaluation", ratio=3)
        for me in added:
            table.add_row(
                _name_for(store, "model", me.model_id),
                _name_for(store, "eval", me.evaluation_id),
            )
        console.print(table)

    if skipped:
        console.print(
            f"[dim]{skipped} already attached, skipped.[/dim]",
        )
    if not added and not skipped:
        console.print("[dim]Nothing to attach.[/dim]")


def _model_evaluations_detach(
    args: argparse.Namespace,
    store: JsonRepository,
    user_store: JsonRepository,
) -> None:
    models = _resolve_models(args, store)
    evaluations = _resolve_evaluations(args, store)

    if not models and not evaluations:
        raise ValueError("specify at least --model/--model-tag or --eval/--eval-tag")

    model_ids = {m.id for m in models} if models else None
    eval_ids = {e.id for e in evaluations} if evaluations else None

    removed: list[DbModelEvaluation] = []

    if model_ids and eval_ids:
        for mid in model_ids:
            for eid in eval_ids:
                removed.extend(
                    user_store.model_evaluations.remove(
                        model_id=mid,
                        evaluation_id=eid,
                    ),
                )
    elif model_ids:
        for mid in model_ids:
            removed.extend(
                user_store.model_evaluations.remove(
                    model_id=mid,
                ),
            )
    elif eval_ids:
        for eid in eval_ids:
            removed.extend(
                user_store.model_evaluations.remove(
                    evaluation_id=eid,
                ),
            )

    if removed:
        user_store.save()
        table = Table(
            title=f"Detached ({len(removed)} removed)",
            show_lines=False,
            width=CLI_WIDTH,
        )
        table.add_column("Model", ratio=2)
        table.add_column("Evaluation", ratio=3)
        for me in removed:
            table.add_row(
                _name_for(store, "model", me.model_id),
                _name_for(store, "eval", me.evaluation_id),
            )
        console.print(table)
    else:
        console.print("[dim]No matching assignments found.[/dim]")


def _name_for(
    store: JsonRepository,
    kind: str,
    id: str,
) -> str:
    try:
        if kind == "model":
            return store.models.get(id).name
        return store.evaluations.get(id).name
    except KeyError:
        return id


# -- run -----------------------------------------------------------


def _register_run(
    subparsers: argparse._SubParsersAction,
) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Run evaluations",
        parents=[_config_parent],
    )
    run_parser.add_argument(
        "model_evaluation_id",
        nargs="?",
        help="Model-evaluation ID to run",
    )
    run_parser.add_argument(
        "--model",
        help="Run all evaluations for this model ID",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: eval_<timestamp>.<format>)",
    )
    run_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "sarif"],
        default="json",
        help="Output format (default: json)",
    )
    run_parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=20,
        help="Max concurrent tasks (default: 20)",
    )
    run_parser.add_argument(
        "--model-tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter by model tag (repeatable)",
    )
    run_parser.add_argument(
        "--eval-tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter by evaluation tag (repeatable)",
    )
    run_parser.add_argument(
        "--fail-under",
        type=_score_threshold,
        metavar="SCORE",
        help=(
            "Exit 5 if any evaluation scores below SCORE (0-1). "
            "Omit to always exit 0 whatever the run scored."
        ),
    )


def _score_threshold(raw: str) -> float:
    """argparse type for `--fail-under`: a score in [0, 1].

    Rejecting here rather than in `handle_run` is deliberate — argparse exits
    2, the code the table reserves for a bad command line, and it does so
    before the run burns a single model call.
    """
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not a number; --fail-under takes a score between 0 and 1",
        ) from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"score threshold must be between 0 and 1, got {value}",
        )
    return value


async def handle_run(
    args: argparse.Namespace,
    store: JsonRepository,
) -> str | None:
    # Imported lazily: the runner and eval core pull in the evaluation
    # framework, which is only needed when a run actually executes.
    from switch_trust.cli.rich_observer import (  # noqa: PLC0415 - deferred import cost
        create_progress,
    )
    from switch_trust.cli.runner import (  # noqa: PLC0415 - deferred import cost
        MAX_CONSECUTIVE_RUN_FAILURES,
        log_run_summary,
        print_results,
        run_cli_evaluation,
        write_output,
    )
    from switch_trust.eval.core.models.model import (  # noqa: PLC0415 - deferred import cost
        track_model_errors,
    )

    model_tag_filter = _parse_tags(args.model_tag)
    eval_tag_filter = _parse_tags(args.eval_tag)

    model_evaluations = _resolve_run_targets(
        store,
        args,
        model_tag_filter,
        eval_tag_filter,
    )
    if not model_evaluations:
        _gate_targets(args, model_evaluations)
        return None

    _check_target_credentials(store, model_evaluations)

    eval_names = [
        store.evaluations.get(me.evaluation_id).name for me in model_evaluations
    ]
    desc_width = min(max((len(n) for n in eval_names), default=20), 40)

    progress = create_progress()

    results: list[CliRunResult] = []
    consecutive_failures = 0
    stopped: CliExit | None = None
    # Every `Model.generate()` failure inside this scope is counted and
    # classified, which is the only way the CLI can tell "the provider
    # rejected your key" from "the run went badly". The engine opens the same
    # scope on the platform path; the CLI never did, so `ErrorKind` existed
    # but nothing here could see it.
    with track_model_errors() as model_errors:
        try:
            progress.start()
            for me in model_evaluations:
                try:
                    result = await run_cli_evaluation(
                        me,
                        store,
                        args.concurrency,
                        progress=progress,
                        desc_width=desc_width,
                    )
                except CliExit as e:
                    # A classified stop from *one* evaluation (a missing judge
                    # credential, an unreadable prompt file, an uninstalled
                    # optional tool) ends the batch — none of the three come
                    # good by trying the next evaluation, and the remaining
                    # ones would only re-pay the spend to fail the same way.
                    # But it must not throw away what already scored: caught
                    # here and re-raised below, the earlier evaluations still
                    # reach `write_output` and the stop carries the artifact.
                    stopped = e
                    break
                results.append(result)
                if result.summary and result.summary.status == EvaluationStatus.ERROR:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_RUN_FAILURES:
                        console.print(
                            f"[red]Aborting: {consecutive_failures} consecutive"
                            f" evaluations failed. Is the target reachable?[/red]",
                        )
                        # No flag is set for this: the shortfall between
                        # `model_evaluations` and `results` *is* the record of
                        # the abort, and `_check_run_completed` reads it there.
                        # A break that forgot to set a flag would go back to
                        # exiting 0 on a suite it never finished.
                        break
                else:
                    consecutive_failures = 0
        finally:
            progress.stop()

    if not results:
        # Nothing scored, so there is no report for the stop to carry.
        if stopped is not None:
            raise stopped
        _gate_targets(args, results)
        return None

    print_results(results)
    log_run_summary(results)

    fmt = getattr(args, "format", "json") or "json"
    ext = fmt if fmt != "json" else "json"
    output_path = args.output or (
        f"eval_{datetime.now().strftime('%Y%m%dT%H%M%S')}.{ext}"
    )
    # Every stop below here runs after the report lands, so it still leaves
    # its artifact, and they run in the documented precedence exit codes
    # 20-39 > 6 > 5.
    write_output(results, args.config, output_path, fmt=fmt)
    # A stop that ended the batch early is why the run has only part of its
    # results, so it is the outcome to report.
    if stopped is not None:
        stopped.output_path = output_path
        raise stopped
    _check_run_completed(results, model_errors, len(model_evaluations), output_path)
    _apply_fail_under(args, results, output_path)
    return output_path


# -- the credential preflight -----------------------------------------
#
# Which provider each keyed SDK adapter talks to, and so whose credentials it
# falls back to when a model's `key` is empty
# (`db/base/models/model_helpers.py::_create_inner_model` omits `api_key=`
# entirely, leaving the SDK to read its own). Only these three types are
# listed because only they have a fallback to check: `ollama` and `litellm`
# need no key of ours, `openai_compatible` defaults to a placeholder one, and
# the agent/HTTP types carry their credential in `headers`, where a missing
# var is already a config error `resolve_env` reports by name.
#
# The credentials themselves are `common/llm_credentials.py`'s, shared with the
# scanner's preflight and the judge's own check. Mapping `ModelType` onto a
# provider is the part that is ours, and all this table does.
_PROVIDER_FOR_MODEL_TYPE: dict[ModelType, str] = {
    ModelType.GEMINI: llm_credentials.PROVIDER_GOOGLE,
    ModelType.OPENAI: llm_credentials.PROVIDER_OPENAI,
    ModelType.ANTHROPIC: llm_credentials.PROVIDER_ANTHROPIC,
}


def _keyless_credential_present(model_type: ModelType) -> bool:
    """Whether this provider will authenticate the model with no API key.

    An API key is the usual credential, not the only one, and refusing a
    target that would have worked is the worse of the two mistakes this
    preflight can make: it is a hard stop with no override, while letting a
    doubtful credential through costs one run that the post-run accumulator
    then classifies properly. So the question asked here is only whether the
    SDK can build a client at all.

    **Gemini**: `GOOGLE_GENAI_USE_VERTEXAI` puts google-genai in Vertex mode,
    where `Client()` fills the project in from ADC and needs no key — the
    "No API key was provided" raise lives in the Developer-API branch it never
    reaches. The *narrow* `vertexai_mode_enabled`, not `google_adc_present`:
    `GOOGLE_APPLICATION_CREDENTIALS` on its own leaves the client in
    Developer-API mode, where it still raises, so it is not a credential here
    — nor, despite what this note used to say, for the scanner, which builds
    the same google-genai client through ADK. `llm_credentials`' module
    docstring carries the detail.

    **Anthropic**: the SDK resolves `ANTHROPIC_AUTH_TOKEN` into an
    `Authorization: Bearer` header, which is how a target behind a gateway is
    configured; it raises for neither credential at construction time.

    OpenAI has no such path — `AsyncOpenAI()` raises when no key resolves — so
    the preflight and the SDK already agree there.
    """
    if model_type is ModelType.GEMINI:
        return llm_credentials.vertexai_mode_enabled()
    if model_type is ModelType.ANTHROPIC:
        return llm_credentials.anthropic_auth_token_present()
    return False


def _check_target_credentials(
    store: JsonRepository,
    model_evaluations: list,
) -> None:
    """Refuse a model under test with no API key, before the run starts.

    The counterpart to `scan_cli._check_configured_model`, for the other
    command. Without it a missing key surfaces from inside the provider SDK as
    a bare `ValueError` while the first evaluation is being built: no model
    call is ever made, so the accumulator `_check_model_credentials` reads
    stays empty and the run exits 1 — which tells a CI gate nothing and
    invites a retry that cannot work.

    A preflight rather than a post-mortem because it costs the user nothing
    here: no evaluation has started, so there is no report to carry and no
    `output_path` on the raise.

    Only *absence* is checked, and only for the keyed SDK providers. A key the
    provider rejects stays the post-run accumulator's job — it takes a call to
    find that out — and a type with no key to miss is not an error. Nor is a
    provider holding a credential that is not an API key at all, which is
    `_keyless_credential_present`'s half of the question.
    """
    for model_id in dict.fromkeys(me.model_id for me in model_evaluations):
        try:
            db_model = store.models.get(model_id)
        except KeyError:
            # Not this check's to report: the run loop raises on the missing
            # model with the context to say which assignment referenced it.
            continue

        provider = _PROVIDER_FOR_MODEL_TYPE.get(db_model.type)
        if provider is None:
            continue
        env_vars = llm_credentials.API_KEY_ENV_VARS[provider]

        try:
            key = resolve_env(db_model.key)
        except ValueError as e:
            # The config names a variable for the key and the variable is not
            # set. Still a missing credential, so still 20 — and naming the
            # variable is the whole fix.
            raise CliExit(
                ExitCode.AUTH_FAILED,
                f"The API key for model {db_model.name!r} could not be "
                f"resolved: {e}. Set it, or put a key in the model's config, "
                f"and re-run.",
            ) from e

        if (
            (key and key.strip())
            or llm_credentials.api_key_env_present(provider)
            or _keyless_credential_present(db_model.type)
        ):
            continue

        raise CliExit(
            ExitCode.AUTH_FAILED,
            f"No API key is configured for model {db_model.name!r}. Set "
            f"{' or '.join(env_vars)}, or put a key in the model's config, "
            f"and re-run.",
        )


# -- classified run failures ------------------------------------------
#
# `ErrorKind` is the eval engine's own taxonomy for why a model call failed,
# and it is richer than the exit table needs. This maps only the kinds whose
# exit code is unambiguous; everything else keeps falling through to the
# catch-all 1 rather than being guessed at. The omissions are deliberate:
#
#   PLAN_LIMIT_EXCEEDED — has a clean target (21) and is simply not in scope
#                here; add with a test rather than in passing.
#   INTERNAL_ERROR / UNKNOWN — ours or unclassified, which is what 1 means.
#
# QUOTA_EXCEEDED is split below rather than mapped directly, the same way
# UNREACHABLE is — see the comment above `_recorded_status_kind`.
#
# INVALID_CONFIG and TIMEOUT are both unambiguous, unlike UNREACHABLE below:
# every status that `classify_model_error` sorts into INVALID_CONFIG
# (INVALID_ARGUMENT/FAILED_PRECONDITION, or any residual 4xx not already
# claimed by auth/timeout/quota/unreachable) means the same thing — the
# endpoint answered and rejected the request shape — so it maps directly, with
# no status sub-split needed. TIMEOUT is the same: a bare `TimeoutError`, a
# 408/504, and google.rpc's `DEADLINE_EXCEEDED` are all the provider call
# giving up, with no further split that would change the diagnosis.
_EXIT_FOR_ERROR_KIND = {
    ErrorKind.AUTH_FAILED: ExitCode.AUTH_FAILED,
    ErrorKind.INVALID_CONFIG: ExitCode.INVALID_CONFIG,
    ErrorKind.TIMEOUT: ExitCode.TIMEOUT,
}

# The two statuses `ModelErrorKey.status` records for `UNREACHABLE` when no
# HTTP status exists at all: `model.py` stamps `HOST_NOT_FOUND` on an NXDOMAIN
# and otherwise records `getattr(exc, "status", None)`, which stringifies to
# the literal `"None"` for a bare connection refusal. Either means the network
# itself is the problem, not the endpoint or its answer, hence exit 60.
# Neither is an HTTP status, which is why they stay here rather than joining
# the shared `StatusKind` table.
_NETWORK_FAILURE_STATUSES = frozenset({HOST_NOT_FOUND, str(None)})


# Three `ErrorKind`s are coarser than the exit table and get sub-split on the
# status `ModelErrorKey` recorded per call. They are the whole reason this
# file reads statuses at all:
#
#   AUTH_FAILED — `classify_model_error` maps both 401 and 403 to it (it is
#       the customer-facing taxonomy, not the exit-code table), so the kind
#       alone can't tell a rejected key (20) from a valid key the provider
#       refused as unpermitted (21, plan/quota/scope).
#   UNREACHABLE — collapses a 404 (the configured endpoint is wrong — fix it,
#       exit 24), a 5xx (the provider's own backend failed — retry it, exit
#       63), "host not found" and a bare connection refusal (retry it, exit
#       60). Hence its absence from `_EXIT_FOR_ERROR_KIND` above.
#   QUOTA_EXCEEDED — collapses an explicit HTTP 429 (a real request-rate cap
#       the provider counted and answered over HTTP — retry it, exit 64) with
#       a provider-declared rate limit that carried no status at all (exit
#       61), which is what a provider reaching `_ERROR_STATUS_KINDS` through
#       its google.rpc string alone leaves behind. Both are equally worth
#       re-running; only the literal 429 gets the more specific code.
#
#       The split is on the *status*, not on the rpc string, so Gemini's
#       `RESOURCE_EXHAUSTED` is exit 64: it arrives as a real 429 (on
#       `APIError.code`), and the string rides `ModelErrorKey.error_status`
#       rather than standing in for a status it does have. It read 61 while
#       `model.py` recorded that string *as* the status — an artifact of the
#       record site, not a distinction this table ever meant to draw.
#
# What each *status* means is not decided here — that is the shared table.
def _recorded_status_kind(status: str | None) -> StatusKind | None:
    """The shared meaning of a status `ModelErrorKey` recorded, as a string.

    The accumulator keys on strings because the values it sees are not all
    HTTP codes — `HOST_NOT_FOUND`, google.rpc names and the literal `"None"`
    share the field. Anything non-numeric has no HTTP meaning and yields
    `None`, which leaves it to the branches above.

    This is a lookup, not a second table. The kinds that used to be spelled
    out here as `"403"`, `"404"`, `"429"` and `status.isdigit() and
    int(status) >= 500` now come from `http_status_kinds`, so the CLI cannot
    disagree with the engine about which statuses mean what — including the
    504 that this function previously had to document as "already claimed by
    TIMEOUT upstream, so it never reaches here".
    """
    if status is None or not status.isdigit():
        return None
    return status_kind(int(status))


def _dominant_status_for(
    model_errors: ModelErrorAccumulator, kind: ErrorKind
) -> str | None:
    """The HTTP status behind most of the run's calls of one `kind`, if any."""
    totals: dict[str, int] = {}
    for key, count in model_errors.counts().items():
        if key.kind is kind:
            totals[key.status] = totals.get(key.status, 0) + count
    if not totals:
        return None
    return max(totals, key=lambda status: totals[status])


def _check_model_credentials(
    model_errors: ModelErrorAccumulator,
    output_path: str,
) -> None:
    """Name the model failure behind a run that produced no score.

    **Only ever called for a run with no verdict** — `_check_run_completed`
    owns that question and this is the "why" it asks for. The accumulator
    cannot answer it on its own: `dominant_kind()` ranks the recorded kinds
    against *each other*, never against the calls that succeeded, so it is
    non-`None` whenever a single call failed anywhere in the run. Every
    failed attempt is recorded at the `Model.generate()` choke point and
    `RetryModel` retries through that same public method, so even a 429 the
    retry recovered is in there. Called unconditionally, this would exit 64
    on a 1000-prompt run where three prompts hit a 429 and recovered — report
    written, every score valid, and a wrapper branching on the 60-79 band
    re-running the lot.

    Given no verdict, the prevailing kind *is* the explanation. A dominant
    `auth_failed` means the credential is wrong or unpermitted, a dominant
    `unreachable` whose calls were all 404s means the configured endpoint is
    wrong, and a dominant `invalid_config` means the provider rejected the
    request shape itself — none of these is fixed by re-running, which is
    exactly what the 20-39 band is for. A dominant `unreachable` with no
    status at all — a bare connection refusal or an unresolvable host — is
    the one case that *is* worth re-running as-is, which is why it lands in
    the 60-79 band instead, and a dominant `unreachable` whose calls were a
    literal 5xx is the same: the provider's own backend failed, not the
    request we sent it. A dominant `quota_exceeded` with a literal `"429"`
    behind it is a real request-rate cap the provider counted and answered
    over HTTP (exit 64) — including Gemini's `RESOURCE_EXHAUSTED`, which is
    one; anything else is the same shape of exception without that status
    (exit 61).

    Returning without raising is normal, not a fallthrough: a kind with no
    entry in the exit table (`internal_error`, `unknown`) leaves the scoreless
    run to `ExitCode.EVAL_INCOMPLETE`, which is the right answer for it.
    """
    kind = model_errors.dominant_kind()
    code = _EXIT_FOR_ERROR_KIND.get(kind)

    if kind is ErrorKind.AUTH_FAILED:
        status_meaning = _recorded_status_kind(_dominant_status_for(model_errors, kind))
        if status_meaning is StatusKind.NOT_ENTITLED:
            code = ExitCode.NOT_ENTITLED
    elif kind is ErrorKind.UNREACHABLE:
        status = _dominant_status_for(model_errors, kind)
        status_meaning = _recorded_status_kind(status)
        if status_meaning is StatusKind.NOT_FOUND:
            code = ExitCode.WRONG_URL
        elif status in _NETWORK_FAILURE_STATUSES:
            code = ExitCode.NETWORK_FAILURE
        elif status_meaning is StatusKind.UPSTREAM:
            code = ExitCode.UPSTREAM_ERROR
    elif kind is ErrorKind.QUOTA_EXCEEDED:
        status_meaning = _recorded_status_kind(_dominant_status_for(model_errors, kind))
        if status_meaning is StatusKind.RATE_CAP:
            code = ExitCode.RESOURCE_EXHAUSTED
        else:
            code = ExitCode.RATE_LIMITED

    if code is None:
        return

    failed = sum(
        count for key, count in model_errors.counts().items() if key.kind is kind
    )
    plural = "s" if failed > 1 else ""
    if code is ExitCode.NOT_ENTITLED:
        message = (
            f"The provider accepted the configured credential but refused it as "
            f"not permitted on {failed} model call{plural}. Check the plan/quota "
            f"for the model under test and re-run."
        )
    elif code is ExitCode.WRONG_URL:
        message = (
            f"The configured endpoint returned 404 Not Found on {failed} model "
            f"call{plural}. Check the URL for the model under test and re-run."
        )
    elif code is ExitCode.INVALID_CONFIG:
        message = (
            f"The provider rejected the request shape on {failed} model "
            f"call{plural}. Check the model configuration (input/output paths, "
            f"payload) and re-run."
        )
    elif code is ExitCode.NETWORK_FAILURE:
        message = (
            f"Could not reach the configured endpoint on {failed} model "
            f"call{plural}. This is often transient — re-run, and check "
            f"connectivity/DNS for the model under test if it persists."
        )
    elif code is ExitCode.UPSTREAM_ERROR:
        message = (
            f"The configured endpoint's backend failed on {failed} model "
            f"call{plural}. This is often transient — re-run."
        )
    elif code is ExitCode.RESOURCE_EXHAUSTED:
        message = (
            f"The provider rate-limited {failed} model call{plural} with "
            f"HTTP 429. This is often transient — wait a moment and re-run."
        )
    elif code is ExitCode.RATE_LIMITED:
        message = (
            f"The provider declared a rate limit on {failed} model "
            f"call{plural}. This is often transient — wait a moment and re-run."
        )
    elif code is ExitCode.TIMEOUT:
        message = (
            f"The provider call timed out on {failed} model "
            f"call{plural}. This is often transient — re-run."
        )
    else:
        message = (
            f"The provider rejected the configured credential on {failed} model "
            f"call{plural}. Check the API key for the model under test and re-run."
        )
    raise CliExit(code, message, output_path=output_path)


# -- completeness, then the --fail-under gate -------------------------
#
# Both are checked in one place, after the report is written, so a job that
# stops still collects its artifact. Every stop they can produce carries the
# output path for the "Results:" footer (see `exit_codes.CliExit`).
#
# The two answer different questions and only one of them is opt-in.
# `_check_run_completed` asks whether the run produced a verdict at all and
# runs unconditionally; `_apply_fail_under` asks whether that verdict clears
# the user's bar and runs only when they set one.
#
# `_check_model_credentials` is not a third question: it is how
# `_check_run_completed` names the cause when the answer is no, which is why
# it is called from inside rather than beside it. That placement is the
# precedence 20-39 > 6 > 5, and it is also the gate — see its docstring for
# why running it on a scored run exits non-zero on a healthy one.
#
# Scope note: a run where *some* evaluations errored and the rest produced a
# score exits 0 (or trips the gate on what scored). That holds however badly
# those evaluations failed — a rejected credential behind 600 of 1000 prompts
# still exits 0 if the other 400 scored. Calling that incomplete needs a
# policy for how much error is too much, which is a product decision, not a
# default worth inventing here, and the per-evaluation errors are already on
# screen.
#
# That note covers a *finished* run, and only a finished run. An evaluation
# that errored was still asked and answered; one the abort in `handle_run`
# skipped was never put to the model at all, so there is no error on screen
# for it and no policy question to settle — the suite the user named is
# simply not the suite that ran. `_check_run_completed` therefore asks two
# things, not one: did anything score, and was every target reached. Both
# have to be yes before a run can exit 0 or have its `--fail-under` gate
# judged, because a gate applied to the fragment that ran would report "the
# bar was met" on the strength of the evaluations that happened to come
# first.

_MAX_NAMED_IN_MESSAGE = 3


def _gate_targets(
    args: argparse.Namespace,
    model_evaluations: list,
) -> None:
    """Refuse a gated run that resolved nothing to evaluate.

    Without `--fail-under` this stays a yellow note and exit 0, as it always has.
    With one, exiting 0 would report "the bar was met" for a run that measured
    nothing — the false green a gate exists to prevent. It is `NO_TARGETS`
    rather than `EVAL_INCOMPLETE` because the cause is a selector or tag filter
    the user has to fix, which is what the 20-39 band means.
    """
    if model_evaluations or getattr(args, "fail_under", None) is None:
        return
    raise CliExit(
        ExitCode.NO_TARGETS,
        f"No evaluations ran, so --fail-under {args.fail_under:.2f} could not be checked.",
    )


def _produced_a_verdict(results: list[CliRunResult]) -> bool:
    """Whether the run scored anything at all.

    `EvaluationSummary.score` is `None` for anything that did not finish, so
    one scored evaluation among errored ones still counts: there is a number
    to report and the user can see what failed beside it.
    """
    return any(r.summary is not None and r.summary.score is not None for r in results)


def _check_run_completed(
    results: list[CliRunResult],
    model_errors: ModelErrorAccumulator,
    planned: int,
    output_path: str,
) -> None:
    """Stop a run that produced no score, or never reached every target.

    This is the one backwards-compatibility break the exit-code work makes on
    purpose, and the defect CHOO-3479 was filed about: a run where every
    evaluation errored used to write its report and exit 0, which is
    indistinguishable from a run where everything passed.

    Deliberately *not* behind `--fail-under`, unlike exit 5. The flags default
    to off so that a verdict the user never asked to be judged cannot become a
    new CI failure — but "no verdict was produced" is not a verdict, it is the
    false green. Gating the fix behind the flag would leave every pipeline
    that has not opted in still reporting a clean run for a total outage,
    which is the whole defect.

    Exit 6 is the fallback, not the first answer: the accumulator usually
    knows *which* failure emptied the run, and `_check_model_credentials`
    raises the specific code for it (20-39 to fix, 60-79 to retry) before we
    settle for "no verdict". Only a kind the exit table has no entry for, or a
    run that scored nothing without any model call failing, reaches the raise
    below.

    Not in the 60-79 retry band itself, even though the cause usually is
    transient: the CLI has already exhausted the retries it does on its own
    (per-prompt timeout/unreachable/quota), so re-running a whole eval is the
    user's call rather than something a wrapper should automate. See
    `ExitCode.EVAL_INCOMPLETE`.

    The second stop is the same code for a different shortfall: `planned`
    evaluations were resolved and fewer of them ran, which only happens when
    the consecutive-failure abort in `handle_run` cut the suite short. One
    early score is enough to satisfy the first question and would otherwise
    carry the whole run to exit 0 — but a target the CLI gave up on before
    reaching is not an evaluation that passed, and reporting it as one is the
    same false green under a different cause.
    """
    if not _produced_a_verdict(results):
        _check_model_credentials(model_errors, output_path)
        raise _no_verdict_exit(results, planned, output_path)

    if len(results) < planned:
        raise _unfinished_suite_exit(results, planned, output_path)


def _skipped_clause(results: list[CliRunResult], planned: int) -> str:
    """`", N never ran"`, or nothing when the suite finished.

    Counted against what `handle_run` resolved rather than carried out of the
    loop, so a future `break` cannot forget to report itself.
    """
    skipped = planned - len(results)
    return f", {skipped} never ran" if skipped > 0 else ""


def _no_verdict_exit(
    results: list[CliRunResult],
    planned: int,
    output_path: str,
) -> CliExit:
    """The bare "nothing scored" stop, for when no specific cause is known."""
    errored = sum(
        1
        for r in results
        if r.summary is not None and r.summary.status == EvaluationStatus.ERROR
    )
    return CliExit(
        ExitCode.EVAL_INCOMPLETE,
        f"No evaluation produced a score ({errored} of {len(results)} "
        f"errored{_skipped_clause(results, planned)}), so this run has no "
        f"verdict to report. Check the per-evaluation errors above and re-run.",
        output_path=output_path,
    )


def _unfinished_suite_exit(
    results: list[CliRunResult],
    planned: int,
    output_path: str,
) -> CliExit:
    """The "it scored, but not on everything you asked for" stop.

    Phrased around the targets that never ran rather than the abort that
    caused it: the abort is already on screen in red, and the fact a wrapper
    needs is that the report covers part of the suite.
    """
    return CliExit(
        ExitCode.EVAL_INCOMPLETE,
        f"The run stopped early with {planned - len(results)} of {planned} "
        f"evaluations never run, so its scores are not a verdict on the "
        f"suite. Check the per-evaluation errors above and re-run.",
        output_path=output_path,
    )


def _apply_fail_under(
    args: argparse.Namespace,
    results: list[CliRunResult],
    output_path: str,
) -> None:
    """Trip the `--fail-under` gate, if one was asked for and the run cleared it.

    `EvaluationSummary.score` is `None` for anything that did not finish, so
    an errored evaluation is never scored as a zero — that would report a
    provider outage as a model that failed the bar, which is the specific
    confusion `EVAL_INCOMPLETE` exists to keep separate.

    The comparison is strictly-below: `--fail-under 0.8` is the bar a score must
    meet, so exactly 0.8 passes.
    """
    threshold = getattr(args, "fail_under", None)
    if threshold is None:
        return

    scored = [
        (_name_of(r), r.summary.score)
        for r in results
        if r.summary is not None and r.summary.score is not None
    ]
    if not scored:
        # Unreachable from `handle_run`, which runs `_check_run_completed`
        # first. Raising rather than returning is what keeps it that way: a
        # bare `return` here would report the bar as met for a run that
        # measured nothing, if the two were ever reordered. It is the bare
        # exit 6 rather than the classified one because naming the cause
        # needs the accumulator, and threading it through a branch nothing
        # reaches would buy a better message on a path that never runs. The
        # planned count is `len(results)` for the same reason: the gate is
        # not told how many targets were resolved, so it claims no skips
        # rather than guessing at them.
        raise _no_verdict_exit(results, len(results), output_path)

    below = sorted(
        ((name, score) for name, score in scored if score < threshold),
        key=lambda pair: pair[1],
    )
    if not below:
        return

    raise CliExit(
        ExitCode.FINDINGS_THRESHOLD,
        f"{len(below)} of {len(scored)} evaluations scored below "
        f"{threshold:.2f}: {_summarise_below(below)}",
        output_path=output_path,
    )


def _name_of(result: CliRunResult) -> str:
    return result.evaluation.get("name") or result.model_evaluation_name or "?"


def _summarise_below(below: list[tuple[str, float]]) -> str:
    """Name the worst offenders, not all of them — this goes in a panel."""
    named = ", ".join(
        f"{name} ({score:.2f})" for name, score in below[:_MAX_NAMED_IN_MESSAGE]
    )
    remaining = len(below) - _MAX_NAMED_IN_MESSAGE
    return f"{named}, +{remaining} more" if remaining > 0 else named


def _filter_by_tags(
    items: list,
    store: JsonRepository,
    model_tags: dict[str, str],
    eval_tags: dict[str, str],
) -> list:
    filtered = []
    for me in items:
        if model_tags:
            try:
                model = store.models.get(me.model_id)
                if not _matches_tags(model.tags, model_tags):
                    continue
            except KeyError:
                continue
        if eval_tags:
            try:
                ev = store.evaluations.get(
                    me.evaluation_id,
                )
                if not _matches_tags(ev.tags, eval_tags):
                    continue
            except KeyError:
                continue
        filtered.append(me)
    return filtered


def _resolve_run_targets(
    store: JsonRepository,
    args: argparse.Namespace,
    model_tags: dict[str, str],
    eval_tags: dict[str, str],
) -> list:
    if args.model_evaluation_id and args.model:
        raise ValueError("specify either a model-evaluation ID or --model, not both")

    if not args.model_evaluation_id and not args.model:
        raise ValueError("specify a model-evaluation ID or --model <model-id>")

    if args.model_evaluation_id:
        me_id = _resolve_id(
            args.model_evaluation_id,
            [me.id for me in store.model_evaluations.list_all()],
        )
        try:
            me = store.model_evaluations.get(me_id)
            items = _filter_by_tags(
                [me],
                store,
                model_tags,
                eval_tags,
            )
            if not items:
                console.print(
                    "[yellow]Model-evaluation filtered out by tag filters.[/yellow]",
                )
            return items
        except KeyError:
            raise ValueError(
                f"model-evaluation {args.model_evaluation_id!r} not found"
            ) from None

    model_id = _resolve_id(
        args.model,
        [m.id for m in store.models.list()],
    )
    try:
        model = store.models.get(model_id)
    except KeyError:
        raise ValueError(f"model {args.model!r} not found") from None

    if model_tags and not _matches_tags(
        model.tags,
        model_tags,
    ):
        console.print(
            f"[yellow]Model {model_id!r} does not match --model-tag filters.[/yellow]",
        )
        return []

    me_view = store.model_evaluations.list_by_model(
        model_id,
        limit=1000,
    )
    items = [item.config for item in me_view.items]
    if eval_tags:
        items = _filter_by_tags(
            items,
            store,
            {},
            eval_tags,
        )
    if not items:
        console.print(
            f"[yellow]No evaluations assigned to model "
            f"{model_id!r} (after tag filtering).[/yellow]",
        )
    return items


# -- formatting helpers -------------------------------------------


def _resolve_id(
    prefix: str,
    all_ids: list[str],
) -> str:
    if prefix in all_ids:
        return prefix
    matches = [full_id for full_id in all_ids if full_id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(
            f"[yellow]Ambiguous ID prefix {prefix!r}, "
            f"matches: {[m[:12] for m in matches]}[/yellow]",
        )
    return prefix


def _fmt_tags(tags: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in tags.items())
