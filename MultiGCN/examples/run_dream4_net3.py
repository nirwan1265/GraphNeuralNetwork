#!/usr/bin/env python
"""
Run MultiGCN on DREAM4 100-gene network 3.

Staged deliberately. Running all 36 algorithms before any dependency is
installed produces 36 failures and no information about which ones matter, so
this starts with the tier that needs nothing beyond pip and widens from there.

    python examples/run_dream4_net3.py --tier python
    python examples/run_dream4_net3.py --tier r
    python examples/run_dream4_net3.py --tier all

Add --gold-standard to get the per-algorithm AUROC/AUPR table.
"""
import argparse
from pathlib import Path

from multigcn import MultiGCN
from multigcn.data.dream4 import summarise, to_time_series

DATA = Path("~/Documents/Github/GraphNeuralNetwork/data/Database.DREAM4"
            ).expanduser()
EXP = DATA / "EXP" / "dream4_100_03_exp.csv"
GS = DATA / "GS" / "dream4_100_03_gs.csv"

# Tier 1: pip only. No R, no Docker, no arboreto - GENIE3/GRNBOOST2 are
# implemented natively on scikit-learn.
PYTHON_TIER = ["GENIE3_RF", "GENIE3_ET", "GRNBOOST2"]

# Tier 2: + R packages, steady-state only (no time-series file needed).
R_TIER = ["ARACNE", "CLR", "MRNET", "MRNETB", "BC3NET", "C3NET",
          "PCIT", "KBOOST", "LEAP", "PPCOR", "GENENET", "BNLEARN"]

# Tier 3: everything that needs the time-series conversion.
TIME_SERIES_TIER = ["DYNGENIE3", "DYNUGENE", "NETWORKBMA", "SCANBMA",
                    "G1DBN", "SWING"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="python",
                    choices=["python", "r", "timeseries", "all"])
    ap.add_argument("--exp", type=Path, default=EXP)
    ap.add_argument("--gold-standard", type=Path, default=GS,
                    help="genes x genes matrix or 3-column edge list; "
                         "format is auto-detected")
    ap.add_argument("--output", type=Path, default=Path("dream4_net3_out"))
    ap.add_argument("--top-x", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()

    if not args.exp.exists():
        raise SystemExit(f"Expression file not found: {args.exp}")

    print(f"Input: {args.exp.name}")
    print(f"  {summarise(args.exp)}\n")

    # The DREAM4 columns are 10 concatenated perturbation time courses, so the
    # temporal structure has to be made explicit for the dynamical methods.
    ts_path = None
    if args.tier in ("timeseries", "all"):
        ts_path = args.output / "dream4_100_03_timeseries.csv"
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        to_time_series(args.exp, ts_path)
        print(f"Time-series view written to {ts_path}\n")

    methods = {
        "python": PYTHON_TIER,
        "r": PYTHON_TIER + R_TIER,
        "timeseries": TIME_SERIES_TIER,
        "all": PYTHON_TIER + R_TIER + TIME_SERIES_TIER,
    }[args.tier]

    # Only pass parameters for algorithms actually in this run. The package
    # now warns rather than crashes on the rest, but keeping this conditional
    # means no spurious warnings on the common tiers.
    extra = {}
    if any(m.startswith("GENIE3") or m == "GRNBOOST2" for m in methods):
        extra["genie3_ntrees"] = 1000
    if "SWING" in methods:
        extra["swing_source_path"] = str(
            Path("~/Documents/Github/SWING").expanduser())
    if "DYNGENIE3" in methods:
        extra["dyngenie3_source_path"] = str(
            Path("~/Documents/Github/dynGENIE3/dynGENIE3_python").expanduser())
    if "DYNUGENE" in methods:
        extra["dynugene_ntree"] = 500

    model = MultiGCN(
        methods=methods,
        top_x=args.top_x,
        epochs=args.epochs,
        iterations=args.iterations,
        output_dir=str(args.output),
        **extra,
    )
    print(model.plan(), "\n")

    # skip_failures keeps a missing R package from killing the whole run; the
    # summary at the end reports exactly which ones dropped out.
    model.infer(args.exp, time_series=ts_path, skip_failures=True)
    model.load()
    print(f"\nLoaded {len(model.matrices)} algorithm matrices: "
          f"{', '.join(sorted(model.matrices))}\n")

    model.fit_predict()
    print(f"\nFused network -> {model.save()}")
    model.save_matrices()

    if args.gold_standard:
        table = model.per_algorithm_scores(args.gold_standard)
        out = args.output / "per_algorithm_scores.csv"
        table.to_csv(out, index=False)
        print(f"\n{table.to_string(index=False)}")
        print(f"\nScores -> {out}")


if __name__ == "__main__":
    main()
