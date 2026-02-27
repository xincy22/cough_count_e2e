from __future__ import annotations

import os
import sys
from pathlib import Path

_TRIAL_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _TRIAL_ROOT.parent.parent
os.environ.setdefault("COUGHCOUNT_WORKSPACE", str(_TRIAL_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import argparse
from pathlib import Path

from coughcount.viz.history import plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize training history from JSON log."
    )
    parser.add_argument("history_file", type=Path, help="Path to history.json file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window in addition to saving it.",
    )
    args = parser.parse_args()

    out_path = plot_training_history(args.history_file, show=args.show)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
