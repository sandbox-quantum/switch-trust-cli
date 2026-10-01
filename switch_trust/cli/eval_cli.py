"""
Subcommands for `switch-trust eval`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from switch_trust.cli.console import CLI_WIDTH, console
from switch_trust.cli.utils import get_switch_trust_config_path
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation

if TYPE_CHECKING:
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
            console.print(
                f"[red]Invalid tag format: {item!r} (expected KEY=VALUE)[/red]",
            )
            sys.exit(1)
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
        CliRunResult,
        log_run_summary,
        print_results,
        run_cli_evaluation,
        write_output,
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
        return None

    eval_names = [
        store.evaluations.get(me.evaluation_id).name for me in model_evaluations
    ]
    desc_width = min(max((len(n) for n in eval_names), default=20), 40)

    progress = create_progress()

    results: list[CliRunResult] = []
    consecutive_failures = 0
    try:
        progress.start()
        for me in model_evaluations:
            result = await run_cli_evaluation(
                me,
                store,
                args.concurrency,
                progress=progress,
                desc_width=desc_width,
            )
            results.append(result)
            if result.summary and result.summary.status == EvaluationStatus.ERROR:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_RUN_FAILURES:
                    console.print(
                        f"[red]Aborting: {consecutive_failures} consecutive"
                        f" evaluations failed. Is the target reachable?[/red]",
                    )
                    break
            else:
                consecutive_failures = 0
    finally:
        progress.stop()

    if not results:
        return None

    print_results(results)
    log_run_summary(results)

    fmt = getattr(args, "format", "json") or "json"
    ext = fmt if fmt != "json" else "json"
    output_path = args.output or (
        f"eval_{datetime.now().strftime('%Y%m%dT%H%M%S')}.{ext}"
    )
    write_output(results, args.config, output_path, fmt=fmt)
    return output_path


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
