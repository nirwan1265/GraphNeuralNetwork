"""
MultiGCN command line interface.

One entry point. Algorithm selection is a single `--algorithms` flag, and every
algorithm parameter follows the same `--<package>-<param>` pattern.

    multigcn list
    multigcn run expression.csv --algorithms all      # everything, containers included
    multigcn run expression.csv --algorithms local    # host runtimes only, no Docker
    multigcn run expression.csv --algorithms r
    multigcn run expression.csv --algorithms GENIE3_RF,SWING,CLR
    multigcn run expression.csv --algorithms all --no-fuse
    multigcn run expression.csv --algorithms all --genie3-ntrees 500

Selection keywords are derived from the registry, never hardcoded, so a newly
registered algorithm joins the right groups automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .params import split_prefixed
from .pipeline import MultiGCN
from .registry import (ALGORITHMS, DataRequirement, Runtime, by_family,
                       by_runtime)

# --------------------------------------------------------------------------
# Algorithm selection
# --------------------------------------------------------------------------

#: Runtimes that execute inside a container rather than on the host.
CONTAINER_RUNTIMES = (Runtime.DOCKER, Runtime.PYTHON_ISOLATED)


def _keyword_groups(lists_dir: Path | None = None) -> dict:
    """Selection keywords, computed from the registry."""
    groups = {
        # 'all' means every registered algorithm, containers included. Use
        # 'local' for the host-only subset - 'all' previously excluded the
        # container runtimes, which silently dropped a third of the catalogue.
        "all": list(ALGORITHMS),
        "everything": list(ALGORITHMS),   # retained as an alias for 'all'
        "local": [n for n, s in ALGORITHMS.items()
                  if s.runtime not in CONTAINER_RUNTIMES],
        "python": [n for n, s in ALGORITHMS.items() if s.runtime is Runtime.PYTHON],
        "r": [n for n, s in ALGORITHMS.items() if s.runtime is Runtime.R],
        "docker": [n for n, s in ALGORITHMS.items()
                   if s.runtime in CONTAINER_RUNTIMES],
        "timeseries": [n for n, s in ALGORITHMS.items()
                       if s.requires is DataRequirement.TIME_SERIES],
        "steadystate": [n for n, s in ALGORITHMS.items()
                        if s.requires is DataRequirement.STEADY_STATE],
    }
    for family, names in by_family().items():
        groups[family] = names
    if lists_dir and lists_dir.is_dir():
        groups["cached"] = sorted(p.stem[len("GRN_"):]
                                  for p in lists_dir.glob("GRN_*.csv"))
    return groups


def resolve_selection(value: str, lists_dir: Path | None = None) -> list:
    """
    '--algorithms' accepts a keyword, an explicit comma-separated list, or a
    mix. Unknown names fail loudly rather than being silently dropped: a
    quietly missing algorithm looks exactly like a bad result downstream.
    """
    groups = _keyword_groups(lists_dir)
    selected, unknown = [], []
    for token in (t.strip() for t in value.split(",") if t.strip()):
        if token in groups:
            selected.extend(groups[token])
        elif token in ALGORITHMS:
            selected.append(token)
        else:
            unknown.append(token)

    if unknown:
        raise SystemExit(
            f"Unknown algorithm or keyword: {', '.join(unknown)}\n"
            f"Keywords: {', '.join(sorted(groups))}\n"
            f"Algorithms: {', '.join(sorted(ALGORITHMS))}"
        )
    # De-duplicate, preserving order.
    return list(dict.fromkeys(selected))


# --------------------------------------------------------------------------
# --<package>-<param> parsing
# --------------------------------------------------------------------------

def parse_algorithm_params(extras: list) -> dict:
    """
    Turn ['--genie3-ntrees', '500', '--bnlearn-R', '20'] into
    {'genie3_ntrees': 500, 'bnlearn_R': 20}.

    Values are left as strings where ambiguous; ParamSpec.validate casts and
    range-checks them before anything runs.
    """
    prefixes = {s.param_prefix for s in ALGORITHMS.values() if s.param_prefix}
    out, i = {}, 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            raise SystemExit(f"Unexpected argument: {token}")
        body = token[2:]
        value = None
        if "=" in body:
            body, value = body.split("=", 1)

        key = body.replace("-", "_")
        # Prefix may itself contain an underscore, so match against the
        # registry rather than splitting on the first separator.
        if split_prefixed(key, list(prefixes)) is None:
            raise SystemExit(
                f"Unknown option --{body}. Algorithm parameters are "
                f"--<package>-<param>, e.g. --genie3-ntrees 500.\n"
                f"Known packages: {', '.join(sorted(prefixes))}"
            )
        if value is None:
            if i + 1 >= len(extras):
                raise SystemExit(f"--{body} needs a value")
            value = extras[i + 1]
            i += 1
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except (TypeError, ValueError):
                continue
        out[key] = value
        i += 1
    return out


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_list(args, extras):
    print(f"{len(ALGORITHMS)} registered algorithm(s)\n")
    for family, names in by_family().items():
        print(f"  {family}")
        for n in names:
            s = ALGORITHMS[n]
            need = "" if s.requires is DataRequirement.STEADY_STATE \
                else f"  <{s.requires.value}>"
            params = (f"  [--{s.param_prefix}-*]" if s.param_prefix else "")
            print(f"    {s.name:<15} {s.runtime.value:<16} "
                  f"{s.library:<22}{need}{params}")
        print()
    print("Selection keywords for --algorithms:")
    for key, names in sorted(_keyword_groups().items()):
        print(f"  {key:<20} {len(names):>2}")


def cmd_run(args, extras):
    algo_params = parse_algorithm_params(extras)
    output = Path(args.output)
    lists_dir = output / "lists"
    methods = resolve_selection(args.algorithms, lists_dir)

    n_container = sum(1 for m in methods
                      if ALGORITHMS[m].runtime in CONTAINER_RUNTIMES)
    if n_container:
        print(f"{n_container} of {len(methods)} selected algorithm(s) run in "
              f"Docker; images are pulled on first use and need a running "
              f"daemon. Pass --algorithms local to skip them.\n")

    cfg = Config.from_file(args.config) if args.config else Config()
    cfg.output_dir = str(output)
    cfg.top_x = args.top_x
    cfg.epochs = args.epochs
    cfg.iterations = args.iterations
    if args.device:
        cfg.device = args.device

    model = MultiGCN(methods=methods, config=cfg, **algo_params)
    print(model.plan(), "\n")
    if args.dry_run:
        return

    start = time.time()

    # Time series: convert automatically when the input looks like DREAM4 and
    # anything selected needs it.
    ts_path = args.time_series
    needs_ts = any(ALGORITHMS[m].requires is DataRequirement.TIME_SERIES
                   for m in methods)
    if needs_ts and ts_path is None:
        from .data.dream4 import is_time_series, to_time_series
        if is_time_series(args.expression):
            ts_path = (output / (Path(args.expression).stem
                                 + "_timeseries.csv")).resolve()
            ts_path.parent.mkdir(parents=True, exist_ok=True)
            to_time_series(args.expression, ts_path)
            print(f"Time-series view -> {ts_path}\n")

    if not args.use_cached:
        model.infer(args.expression, time_series=ts_path,
                    skip_failures=not args.strict)

    model.load(lists_dir, expression_file=args.expression)
    print(f"\n{len(model.matrices)} algorithm matrices: "
          f"{', '.join(sorted(model.matrices))}")

    if args.no_fuse:
        print(f"\nInference only. Edge lists -> {lists_dir}")
        print(f"elapsed {time.time() - start:.0f}s")
        return

    model.fit_predict()
    print(f"\nFused network -> {model.save()}")
    model.save_matrices()

    if args.gold_standard:
        table = model.per_algorithm_scores(args.gold_standard)
        scores_path = output / "scores.csv"
        table.to_csv(scores_path, index=False)

        inputs = table[table["stage"] == "input"]
        fused = table[table["stage"] == "fused"].iloc[0]
        best = inputs.iloc[0]
        print(f"\n{'algorithm':<24}{'AUROC':>10}{'AUPR':>12}")
        print("-" * 46)
        for _, r in inputs.iterrows():
            print(f"{r['method']:<24}{r['auroc']:>10.4f}{r['aupr']:>12.4f}")
        print("-" * 46)
        print(f"{'MultiGCN (fused)':<24}{fused['auroc']:>10.4f}"
              f"{fused['aupr']:>12.4f}")
        print(f"\nvs best input ({best['method']}): "
              f"{fused['aupr'] - best['aupr']:+.4f} AUPR")
        print(f"vs mean input: {fused['aupr'] - inputs['aupr'].mean():+.4f} AUPR")
        print(f"scores -> {scores_path}")

    print(f"\nelapsed {time.time() - start:.0f}s")


def main():
    ap = argparse.ArgumentParser(
        prog="multigcn",
        description="GRN inference by combining many algorithms through a GCN.",
        epilog="Algorithm parameters use --<package>-<param>, "
               "e.g. --genie3-ntrees 500 --bnlearn-R 20. "
               "Run `multigcn list` to see them all.")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show algorithms and selection keywords") \
        .set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="Run the pipeline")
    p.add_argument("expression", help="Expression CSV (genes x samples)")
    p.add_argument("-a", "--algorithms", default="all",
                   help="Keyword (all, local, r, python, docker, timeseries, "
                        "steadystate, cached, or a family) and/or explicit "
                        "names, comma separated. 'all' includes the Docker "
                        "algorithms and needs a running daemon; 'local' is "
                        "the host-only subset. Default: all")
    p.add_argument("-o", "--output", default="multigcn_output")
    p.add_argument("-g", "--gold-standard",
                   help="genes x genes matrix or 3-column edge list")
    p.add_argument("-c", "--config", help="JSON/YAML config file")
    p.add_argument("--time-series", type=Path,
                   help="Explicit time-series CSV (first column 'time'). "
                        "Auto-generated for DREAM4 input when needed.")
    p.add_argument("--no-fuse", action="store_true",
                   help="Run inference only; skip the GCN and fusion")
    p.add_argument("--use-cached", action="store_true",
                   help="Skip inference; use existing edge lists")
    p.add_argument("--strict", action="store_true",
                   help="Stop on the first algorithm failure")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan and exit")
    p.add_argument("--top-x", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--device")
    p.set_defaults(func=cmd_run)

    args, extras = ap.parse_known_args()
    if args.command == "list" and extras:
        raise SystemExit(f"Unexpected argument: {extras[0]}")
    args.func(args, extras)


if __name__ == "__main__":
    main()
