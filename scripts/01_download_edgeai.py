from __future__ import annotations

from coughcount.paths import ProjectPaths as P
from coughcount.data.edgeai import ensure_edgeai_downloaded


def main():
    root = ensure_edgeai_downloaded(P.edgeai_raw)
    print(f"EdgeAI dataset ready at: {root}")


if __name__ == "__main__":
    main()
