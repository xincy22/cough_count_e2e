from __future__ import annotations

from coughcount.paths import ProjectPaths as P
from coughcount.data.edgeai import ensure_edgeai_downloaded, build_manifest


def main():
    public_root = ensure_edgeai_downloaded(P.edgeai_raw)
    df = build_manifest(public_root, P.edgeai_manifest)

    total = int(df.loc[df["class"] == "cough", "cough_count"].sum())
    print(f"Saved manifest: {P.edgeai_manifest} (rows={len(df)})")
    print(f"Total cough events:", total)


if __name__ == "__main__":
    main()
