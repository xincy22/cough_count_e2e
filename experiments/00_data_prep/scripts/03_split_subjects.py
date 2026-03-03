from __future__ import annotations

from coughcount.paths import ProjectPaths as P
from coughcount.data.splits import make_holdout_subject_split


def main():
    splits = make_holdout_subject_split(
        P.edgeai_manifest_csv,
        P.edgeai_splits_json,
        seed=42,
        val_subjects=2,
        test_subjects=2,
    )
    print({k: len(v) for k, v in splits.items() if k != "meta"})
    print("Subject splits saved to", P.edgeai_splits_json)


if __name__ == "__main__":
    main()
