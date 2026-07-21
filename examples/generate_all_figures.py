"""Regenerate every documentation figure in one command.

Runs each plotting example's ``main()`` in turn, so the images embedded in the
README and docs/ (which are committed to the repo) can be refreshed after an
indicator changes. Each example also saves a copy into docs/assets/.

Run with:
    uv run python examples/generate_all_figures.py
"""

import sys
from pathlib import Path

# Allow running from any working directory by importing the sibling scripts.
sys.path.insert(0, str(Path(__file__).parent))

from plot_api_family_figures import main as api_family_main  # noqa: E402
from plot_classic_indicators import main as classic_main  # noqa: E402
from plot_entropy import main as entropy_main  # noqa: E402
from plot_liquidity import main as liquidity_main  # noqa: E402
from plot_new_indicators import main as new_indicators_main  # noqa: E402
from plot_regime_conditional_signal import (  # noqa: E402
    main as regime_conditional_main,
)
from plot_regime_dashboard import main as regime_main  # noqa: E402
from plot_trend_volume import main as trend_volume_main  # noqa: E402

FIGURES = {
    "classic indicators": classic_main,
    "trend & volume": trend_volume_main,
    "liquidity & microstructure": liquidity_main,
    "regime dashboard": regime_main,
    "new indicators": new_indicators_main,
    "entropy": entropy_main,
    "regime-conditional composite signal": regime_conditional_main,
    "API reference family figures": api_family_main,
}


def main() -> None:
    for name, fn in FIGURES.items():
        print(f"\n=== {name} ===")
        fn()
    print(f"\nRegenerated {len(FIGURES)} figures.")


if __name__ == "__main__":
    main()
