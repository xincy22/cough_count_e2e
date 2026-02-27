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


from coughcount.paths import ProjectPaths as P
from coughcount.data.edgeai import ensure_edgeai_downloaded, build_manifest


def main():
    public_root = ensure_edgeai_downloaded(P.edgeai_raw)
    df = build_manifest(public_root, P.edgeai_manifest_csv)

    total = int(df.loc[df["class"] == "cough", "cough_count"].sum())
    print(f"Saved manifest: {P.edgeai_manifest_csv} (rows={len(df)})")
    print(f"Total cough events:", total)


if __name__ == "__main__":
    main()
