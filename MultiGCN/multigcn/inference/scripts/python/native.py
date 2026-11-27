"""
Dispatcher for MultiGCN's native Python algorithms.

Called by PythonRunner like every other wrapper, so native algorithms need no
special handling anywhere in the pipeline:

    native.py <expression_csv> <out_dir> <ALGORITHM> [time_series_csv] [params_json]

The optional trailing arguments are told apart by content: params JSON starts
with '{', anything else is the time-series path.
"""
import json
import sys
from pathlib import Path

# Make the package importable even when run straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from multigcn.inference.native import NATIVE, run_to_file  # noqa: E402


def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: native.py <expression_csv> <out_dir> <ALGORITHM> "
                 "[time_series_csv] [params_json]")
    expression, out_dir, name = sys.argv[1], sys.argv[2], sys.argv[3]
    time_series, params = None, {}
    for arg in sys.argv[4:]:
        if arg.lstrip().startswith("{"):
            params = json.loads(arg)
        elif arg:
            time_series = arg

    if name not in NATIVE:
        sys.exit(f"Unknown native algorithm {name!r}. Available: {', '.join(NATIVE)}")

    path = run_to_file(name, out_dir, expression=expression,
                       time_series=time_series, **params)
    print(f"wrote {sum(1 for _ in open(path))} edges to {path}")


if __name__ == "__main__":
    main()
