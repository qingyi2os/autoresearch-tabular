import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1337
TIME_BUDGET = 60.0
PROSPER_DATASET_FILENAME = "prosper_full_dataset.parquet.gzip"
PROSPER_TARGET = "target"
SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
OOT_SPLIT = "oot"
SPLIT_NAMES = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, OOT_SPLIT)
PROVIDED_SPLIT_SOURCE = "provided_column"
DATASET_CACHE_FILENAME = "dataset.npz"
META_FILENAME = "meta.json"


def default_cache_dir() -> Path:
    root = os.environ.get("AUTORESEARCH_TABULAR_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "autoresearch-tabular"


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative examples.")

    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    concordant = 0.0
    negatives_seen = 0
    i = 0
    while i < len(scores):
        j = i
        pos = 0
        neg = 0
        while j < len(scores) and scores[j] == scores[i]:
            if labels[j] == 1:
                pos += 1
            else:
                neg += 1
            j += 1
        concordant += pos * (negatives_seen + 0.5 * neg)
        negatives_seen += neg
        i = j

    return concordant / (n_pos * n_neg)


def load_prosper_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    dataset_path = dataset_path or (Path(__file__).resolve().parent / PROSPER_DATASET_FILENAME)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prosper dataset not found at {dataset_path}")
    frame = pd.read_parquet(dataset_path)
    frame = frame.dropna(how="all").dropna(axis=1, how="all").copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def load_modeling_frame(
    dataset_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    frame = load_prosper_dataset(dataset_path=dataset_path)
    if PROSPER_TARGET not in frame.columns:
        raise ValueError(f"Target column {PROSPER_TARGET!r} not found.")

    y_series = pd.to_numeric(frame[PROSPER_TARGET], errors="coerce")
    frame = frame.loc[y_series.notna()].copy().reset_index(drop=True)
    y = y_series.loc[y_series.notna()].astype(np.int64).to_numpy()
    excluded = {PROSPER_TARGET, SPLIT_COLUMN}
    x_frame = frame.drop(columns=list(excluded & set(frame.columns))).copy().reset_index(drop=True)
    return frame, x_frame, y, PROSPER_TARGET


def split_indices_from_column(split_values: pd.Series) -> dict[str, np.ndarray]:
    normalized = split_values.astype(str).str.strip().str.lower()
    return {split_name: np.flatnonzero(normalized == split_name) for split_name in SPLIT_NAMES}


def prepare_dataset(
    cache_dir: Path | None = None,
    dataset_path: Path | None = None,
) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame, x_frame, _, target_column = load_modeling_frame(dataset_path=dataset_path)
    split_indices = split_indices_from_column(frame[SPLIT_COLUMN])
    np.savez_compressed(
        cache_dir / DATASET_CACHE_FILENAME,
        **{f"{split_name}_idx": split_indices[split_name] for split_name in SPLIT_NAMES},
        feature_names=np.asarray(x_frame.columns.tolist(), dtype=object),
        feature_dtypes=np.asarray([str(dtype) for dtype in x_frame.dtypes], dtype=object),
    )

    meta = {
        "task": "binary_classification",
        "model": "xgboost",
        "metric": "val_auc",
        "rows": int(len(frame)),
        "features": int(len(x_frame.columns)),
        "target": target_column,
        "cache_dir": str(cache_dir),
        "dataset_path": str(dataset_path),
        "source": "prosper",
        "split_source": PROVIDED_SPLIT_SOURCE,
        "splits": {split_name: int(len(split_indices[split_name])) for split_name in SPLIT_NAMES},
    }
    (cache_dir / META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_dataset(cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    dataset_cache_path = cache_dir / DATASET_CACHE_FILENAME
    meta_path = cache_dir / META_FILENAME
    meta = json.loads(meta_path.read_text())
    with np.load(dataset_cache_path, allow_pickle=True) as dataset:
        split_indices = {
            split_name: dataset[f"{split_name}_idx"].astype(np.int64)
            for split_name in SPLIT_NAMES
        }
        feature_names = dataset["feature_names"].tolist()
        feature_dtypes = dataset["feature_dtypes"].tolist()
    _, x_frame, y, target_column = load_modeling_frame(dataset_path=Path(meta["dataset_path"]))
    return {
        "frame_train": x_frame.iloc[split_indices[TRAIN_SPLIT]].reset_index(drop=True),
        "y_train": y[split_indices[TRAIN_SPLIT]].astype(np.int64),
        "frame_val": x_frame.iloc[split_indices[VAL_SPLIT]].reset_index(drop=True),
        "y_val": y[split_indices[VAL_SPLIT]].astype(np.int64),
        "frame_test": x_frame.iloc[split_indices[TEST_SPLIT]].reset_index(drop=True),
        "y_test": y[split_indices[TEST_SPLIT]].astype(np.int64),
        "frame_oot": x_frame.iloc[split_indices[OOT_SPLIT]].reset_index(drop=True),
        "y_oot": y[split_indices[OOT_SPLIT]].astype(np.int64),
        "feature_names": feature_names,
        "feature_dtypes": feature_dtypes,
        "target_column": target_column,
        "split_source": meta["split_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the autoresearch-glm dataset.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    args = parser.parse_args()

    meta = prepare_dataset(
        cache_dir=args.cache_dir,
        dataset_path=args.dataset_path,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
