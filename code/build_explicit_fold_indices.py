"""Export the exact outer and inner CV memberships used by the fixed-k search.

The completed fixed-k search records hashes for every inner fold.  This script
reconstructs the corresponding memberships from the released split labels and
the recorded seeds, then verifies the hashes against the completed-search CSV.
Indices are the stable ``Original_Index`` values from the 452-record dataset.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold


MASTER_SEED = 42
OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 5
FEATURE_COUNTS = range(3, 18)


def index_hash(ids: list[int]) -> str:
    canonical = ",".join(str(int(value)) for value in sorted(ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in str(value).split(";") if item != ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed_retained_dataset_452.csv"),
    )
    parser.add_argument(
        "--outer-provenance",
        type=Path,
        default=Path("data/outer_fold_indices_explicit.csv"),
    )
    parser.add_argument(
        "--completed-inner-results",
        type=Path,
        default=Path("search_inputs/fixed_k_completed_search/fixed_k_candidate_inner_fold_results.csv.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inner_fold_indices_explicit.csv"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    required = {"Original_Index", "Data_Split"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    proper = data.loc[data["Data_Split"].eq("Proper_Training"), "Original_Index"].astype(int).tolist()
    if len(proper) != 270:
        raise ValueError(f"Expected 270 proper-training rows, found {len(proper)}")

    outer = pd.read_csv(args.outer_provenance)
    expected_outer = {"Outer_ID", "Repetition", "Fold", "Outer_Train_Indices", "Outer_Validation_Indices"}
    missing = expected_outer.difference(outer.columns)
    if missing:
        raise ValueError(f"Outer provenance is missing required columns: {sorted(missing)}")

    completed = pd.read_csv(args.completed_inner_results, compression="infer")
    expected_inner = {
        "Outer_ID", "Feature_Count", "Inner_Fold",
        "Inner_Training_Index_Hash", "Inner_Validation_Index_Hash",
    }
    missing = expected_inner.difference(completed.columns)
    if missing:
        raise ValueError(f"Completed inner results are missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, outer_row in outer.sort_values(["Repetition", "Fold"]).iterrows():
        outer_id = str(outer_row["Outer_ID"])
        outer_train = parse_ids(outer_row["Outer_Train_Indices"])
        outer_validation = parse_ids(outer_row["Outer_Validation_Indices"])
        if len(outer_train) != 216 or len(outer_validation) != 54:
            raise ValueError(f"Unexpected outer sizes for {outer_id}")
        if set(outer_train) | set(outer_validation) != set(proper):
            raise ValueError(f"Outer fold {outer_id} does not partition the proper-training set")
        if set(outer_train) & set(outer_validation):
            raise ValueError(f"Outer fold {outer_id} has overlapping memberships")

        outer_no = (int(outer_row["Repetition"]) - 1) * OUTER_FOLDS + int(outer_row["Fold"])
        outer_seed = MASTER_SEED + 1000 + outer_no
        # run_one_fixed_k uses the outer-training order produced by RepeatedKFold.
        # The explicit provenance stores sorted IDs; RepeatedKFold emits the same
        # order here, and the hash verification below guards against drift.
        for k in FEATURE_COUNTS:
            inner_seed = MASTER_SEED + 10000 + outer_seed * 100 + int(k)
            inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed)
            index_frame = pd.DataFrame(index=outer_train)
            for inner_fold, (train_pos, validation_pos) in enumerate(inner_cv.split(index_frame), 1):
                inner_train = [outer_train[pos] for pos in train_pos]
                inner_validation = [outer_train[pos] for pos in validation_pos]
                match = completed.loc[
                    completed["Outer_ID"].eq(outer_id)
                    & completed["Feature_Count"].eq(int(k))
                    & completed["Inner_Fold"].eq(int(inner_fold))
                ]
                if match.empty:
                    raise ValueError(f"No completed-search hash found for {outer_id}, k={k}, fold={inner_fold}")
                expected_train = set(match["Inner_Training_Index_Hash"].astype(str))
                expected_validation = set(match["Inner_Validation_Index_Hash"].astype(str))
                observed_train = index_hash(inner_train)
                observed_validation = index_hash(inner_validation)
                if observed_train not in expected_train or observed_validation not in expected_validation:
                    raise ValueError(f"Hash mismatch for {outer_id}, k={k}, fold={inner_fold}")
                rows.append(
                    {
                        "Outer_ID": outer_id,
                        "Repetition": int(outer_row["Repetition"]),
                        "Outer_Fold": int(outer_row["Fold"]),
                        "Outer_Train_Count": len(outer_train),
                        "Outer_Validation_Count": len(outer_validation),
                        "Feature_Count": int(k),
                        "Inner_Fold": inner_fold,
                        "Inner_Training_Count": len(inner_train),
                        "Inner_Validation_Count": len(inner_validation),
                        "Inner_CV_Seed": inner_seed,
                        "Inner_Training_Index_Hash": observed_train,
                        "Inner_Validation_Index_Hash": observed_validation,
                        "Inner_Training_Indices": ";".join(str(x) for x in sorted(inner_train)),
                        "Inner_Validation_Indices": ";".join(str(x) for x in sorted(inner_validation)),
                    }
                )

    result = pd.DataFrame(rows)
    expected_rows = OUTER_REPEATS * OUTER_FOLDS * len(list(FEATURE_COUNTS)) * INNER_FOLDS
    if len(result) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} explicit inner-fold rows, found {len(result)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} explicit inner-fold records to {args.output}")
    print("All reconstructed inner-fold hashes match the completed fixed-k search.")


if __name__ == "__main__":
    main()
