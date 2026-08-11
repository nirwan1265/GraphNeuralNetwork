"""Command-line interface: multigcn ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .pipeline import MultiGCN
from .registry import ALGORITHMS, by_family, by_runtime


def cmd_list(args):
    print(f"{len(ALGORITHMS)} registered algorithm(s)\n")
    for family, names in by_family().items():
        print(f"  {family}:")
        for n in names:
            s = ALGORITHMS[n]
            print(f"    {s.name:<14} {s.runtime.value:<16} {s.library}")
        print()
    print("Runtimes:")
    for rt, names in by_runtime().items():
        print(f"  {rt.value:<16} {len(names):>2}  {', '.join(names)}")


def cmd_plan(args):
    model = MultiGCN(methods=args.methods, all_methods=args.all)
    print(model.plan())


def cmd_run(args):
    cfg = Config.from_file(args.config) if args.config else Config()
    if args.top_x is not None:
        cfg.top_x = args.top_x
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.output:
        cfg.output_dir = args.output

    model = MultiGCN(methods=args.methods, all_methods=args.all, config=cfg)
    print(model.plan())

    if not args.lists_dir:
        model.infer(args.expression, skip_failures=args.skip_failures)
        model.load()
    else:
        model.load(args.lists_dir, expression_file=args.expression)

    for prior in args.prior or []:
        name, path = prior.split("=", 1)
        model.add_prior(name, path)

    model.fit_predict()
    out = model.save()
    print(f"\nFused network -> {out}")

    if args.gold_standard:
        res = model.evaluate(args.gold_standard, k=args.tukey_k)
        print("\nResults:")
        for k, v in res.items():
            print(f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
        (Path(cfg.output_dir) / "metrics.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="multigcn", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered algorithms").set_defaults(func=cmd_list)

    p = sub.add_parser("plan", help="Show what would run, and whether it can")
    p.add_argument("-m", "--methods", nargs="+")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="Full pipeline")
    p.add_argument("expression", help="Expression CSV (genes x samples)")
    p.add_argument("-m", "--methods", nargs="+")
    p.add_argument("--all", action="store_true")
    p.add_argument("-o", "--output")
    p.add_argument("-c", "--config", help="JSON/YAML config file")
    p.add_argument("--lists-dir", help="Skip inference; use existing GRN_*.csv here")
    p.add_argument("--prior", action="append", metavar="NAME=PATH",
                   help="Add a curated network as an extra input channel")
    p.add_argument("--top-x", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--gold-standard")
    p.add_argument("--tukey-k", type=float, default=None)
    p.add_argument("--skip-failures", action="store_true")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
