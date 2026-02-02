from __future__ import annotations

from coughcount.paths import ProjectPaths as P
from coughcount.data.splits import make_subject_splits


def main():
    splits = make_subject_splits(
        P.edgeai_manifest_csv,
        P.edgeai_splits_json,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    print({k: len(v) for k, v in splits.items() if k != "meta"})
    print("Subject splits saved to", P.edgeai_splits_json)


if __name__ == "__main__":
    main()
