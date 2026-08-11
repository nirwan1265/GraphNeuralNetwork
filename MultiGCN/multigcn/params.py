"""
Per-algorithm parameter declaration, namespacing and validation.

User-facing parameters are flat and prefixed with the originating package:

    MultiGCN(methods=[...], genie3_ntrees=1000, dyngenie3_tree_method="ET",
             bnlearn_algorithm="tabu", bnlearn_R=200)

The prefix is the *package*, not the algorithm, so GENIE3_RF / GENIE3_ET /
GRNBOOST2 all share the `genie3_` namespace - they are one implementation with
one set of tunables.

Parameters are validated against the declared spec before anything runs. An
unknown or out-of-range value fails immediately rather than being silently
dropped by a wrapper script, because a silently ignored parameter produces a
result that looks like it honoured your setting and did not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of one algorithm."""
    name: str                       # name in the underlying package
    default: Any
    type: type = str
    choices: Optional[Sequence[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    help: str = ""

    def validate(self, value: Any, qualified: str) -> Any:
        if self.type is bool:
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes")
            value = bool(value)
        elif self.type in (int, float) and not isinstance(value, str):
            try:
                value = self.type(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{qualified} must be {self.type.__name__}, got {value!r}")

        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"{qualified} must be one of {list(self.choices)}, got {value!r}")
        if self.minimum is not None and isinstance(value, (int, float)) \
                and value < self.minimum:
            raise ValueError(f"{qualified} must be >= {self.minimum}, got {value}")
        if self.maximum is not None and isinstance(value, (int, float)) \
                and value > self.maximum:
            raise ValueError(f"{qualified} must be <= {self.maximum}, got {value}")
        return value


class ParamError(ValueError):
    pass


def split_prefixed(key: str, prefixes: Sequence[str]) -> Optional[Tuple[str, str]]:
    """
    'genie3_ntrees' -> ('genie3', 'ntrees'), matching the longest known prefix
    first so 'dyngenie3_' is not mistaken for 'genie3_'.
    """
    for prefix in sorted(prefixes, key=len, reverse=True):
        if key.startswith(prefix + "_"):
            return prefix, key[len(prefix) + 1:]
    return None


def route(kwargs: Dict[str, Any], specs) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Split user kwargs into (per-prefix algorithm params, leftover config kwargs).

    Returns leftovers untouched so the caller can apply them to Config.
    """
    prefixes = {s.param_prefix for s in specs if s.param_prefix}
    by_prefix: Dict[str, Dict[str, Any]] = {}
    leftover: Dict[str, Any] = {}

    # Map prefix -> the declared params for that prefix
    declared: Dict[str, Dict[str, ParamSpec]] = {}
    for s in specs:
        if s.param_prefix:
            declared.setdefault(s.param_prefix, {}).update(s.params)

    for key, value in kwargs.items():
        hit = split_prefixed(key, list(prefixes))
        if hit is None:
            leftover[key] = value
            continue
        prefix, pname = hit
        allowed = declared.get(prefix, {})
        if pname not in allowed:
            raise ParamError(
                f"Unknown parameter {key!r}. "
                f"{prefix} accepts: "
                f"{', '.join(f'{prefix}_{p}' for p in sorted(allowed)) or '(none)'}"
            )
        by_prefix.setdefault(prefix, {})[pname] = \
            allowed[pname].validate(value, key)

    return by_prefix, leftover


def resolved_for(spec, user_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Defaults for this algorithm, overridden by anything the user supplied."""
    out = {name: ps.default for name, ps in spec.params.items()}
    out.update(user_params.get(spec.param_prefix, {}))
    return out


def describe(spec) -> str:
    if not spec.params:
        return f"{spec.name}: no tunable parameters exposed."
    lines = [f"{spec.name} (prefix `{spec.param_prefix}_`):"]
    for pname, ps in sorted(spec.params.items()):
        bits = [f"default={ps.default!r}"]
        if ps.choices:
            bits.append(f"choices={list(ps.choices)}")
        if ps.minimum is not None:
            bits.append(f"min={ps.minimum}")
        lines.append(f"  {spec.param_prefix}_{pname:<16} {'; '.join(bits)}")
        if ps.help:
            lines.append(f"      {ps.help}")
    return "\n".join(lines)
