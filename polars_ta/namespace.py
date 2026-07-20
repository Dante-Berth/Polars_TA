"""Native Polars expression namespace: ``pl.col("close").ta.<indicator>(...)``.

Registering a ``.ta`` namespace on ``pl.Expr`` lets every indicator be called
as a method on the expression that supplies its *primary* price input, instead
of passing a column name into a free function::

    df.with_columns(
        pl.col("close").ta.rsi(14).alias("rsi"),
        pl.col("high").ta.average_true_range("low", "close").alias("atr"),
    )

The calling expression is bound to the indicator's **first positional
argument** — which is ``close`` for most indicators and ``high`` for the
high-anchored ones (``stoch``, ``average_true_range``, ``aroon_up``, ...). Every
remaining input column is passed as an ordinary argument, exactly as the free
function takes it, so nothing about the underlying math or signatures changes:
the namespace is a thin dispatch layer over the same functions in
:mod:`polars_ta.momentum`, :mod:`polars_ta.trend`, and friends. Because each
method returns a plain ``pl.Expr``, ``.over("symbol")`` and the streaming engine
work through the namespace unchanged.

The free-function API is unaffected — this is purely additive. A handful of
functions are intentionally **not** exposed here (``cross_sectional_zscore`` /
``cross_sectional_rank`` take a generic value column rather than a price series,
and ``regime_conditional_signal`` arbitrates between two pre-built signal
expressions); call those as free functions.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import polars as pl

from polars_ta import (
    microstructure,
    momentum,
    others,
    quant,
    trend,
    volatility,
    volume,
)

# Modules whose free functions take a price/volume series as their first
# positional argument and return a single ``pl.Expr``. Order matters only for
# the (rare) case of a duplicate name — first module wins — which does not
# occur across these modules today. ``calendar`` is deliberately excluded: its
# functions take a timestamp column, not a price series, so anchoring them to a
# price expression would be misleading — call those as free functions.
_SOURCE_MODULES = (momentum, trend, volatility, volume, others, quant, microstructure)

# Functions to skip: their first positional argument is *not* the calling
# expression's price series, so binding ``self._expr`` to it would be
# nonsensical. Kept as an explicit deny-list (rather than an allow-list) so
# newly added price-series indicators are exposed automatically.
_SKIP: frozenset[str] = frozenset(
    {
        # Cross-sectional helpers operate on a generic value column applied with
        # .over(timestamp); there is no single "price" expr to anchor them to.
        "cross_sectional_zscore",
        "cross_sectional_rank",
        # Arbitrates between two already-computed signal expressions.
        "regime_conditional_signal",
    }
)


def _make_method(func: Callable[..., pl.Expr]) -> Callable[..., pl.Expr]:
    """Wrap a free indicator function into a namespace method that supplies the
    calling expression as the function's first positional argument."""

    def method(self: TAExpr, *args: object, **kwargs: object) -> pl.Expr:
        return func(self._expr, *args, **kwargs)

    method.__name__ = func.__name__
    method.__qualname__ = f"TAExpr.{func.__name__}"
    method.__doc__ = func.__doc__
    # Preserve the original signature for tooling/tab-completion, minus the
    # first (price) parameter, which the namespace now supplies implicitly.
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        method.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            + params[1:]
        )
    except (ValueError, TypeError):  # pragma: no cover - builtins have no sig
        pass
    return method


def _accepts_expr_first_arg(func: Callable[..., pl.Expr]) -> bool:
    """True if ``func`` honors the str-or-Expr convention on its first argument.

    Every indicator in this library coerces its first parameter with the
    ``x if isinstance(x, str) else x`` guard, so binding a bound expression
    there is always valid. This probe exists so the test suite can *assert*
    that invariant holds for every registered indicator — catching a future
    indicator that regresses to an unconditional ``pl.col(...)`` (which only
    accepts a column name) before it ships as a namespace method that raises.

    It builds the expression with a placeholder expr bound to the first
    argument and every remaining *required* positional argument filled with a
    plausible column name; a ``TypeError`` from ``pl.col`` means the function
    rejects an ``Expr`` there.
    """
    params = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    extra = [p.name for p in params[1:] if p.default is inspect.Parameter.empty]
    try:
        func(pl.col("_probe"), *extra)
    except TypeError:
        return False
    return True


def _collect_indicators() -> dict[str, Callable[..., pl.Expr]]:
    """Gather ``{name: function}`` for every namespace-eligible indicator.

    Eligible = a public function defined in one of the source modules, not in
    the deny-list, taking at least one positional argument (the price series).
    The whole library honors the str-or-Expr convention on that first argument,
    so binding the calling expression to it is always valid; the
    :func:`_accepts_expr_first_arg` probe (asserted in the tests) guards against
    a future regression rather than filtering here.
    """
    methods: dict[str, Callable[..., pl.Expr]] = {}
    for module in _SOURCE_MODULES:
        for name, obj in vars(module).items():
            if name.startswith("_") or name in _SKIP:
                continue
            if not inspect.isfunction(obj):
                continue
            if obj.__module__ != module.__name__:
                continue
            methods[name] = obj
    return methods


@pl.api.register_expr_namespace("ta")
class TAExpr:
    """The ``.ta`` accessor on a Polars expression.

    See the module docstring for the calling convention. Every method here is
    generated from the corresponding free function at import time, so the
    namespace can never drift out of sync with the indicator implementations.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr


# Attach one method per indicator. Done at import time so the whole namespace
# exists as soon as ``polars_ta`` is imported.
_INDICATORS = _collect_indicators()
for _name, _func in _INDICATORS.items():
    setattr(TAExpr, _name, _make_method(_func))

#: Sorted names of every indicator reachable via ``pl.col(...).ta.<name>()``.
INDICATOR_NAMES: tuple[str, ...] = tuple(sorted(_INDICATORS))
