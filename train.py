import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from prepare import TIME_BUDGET, auc_score, load_dataset

RANDOM_SEED = 1337
EARLY_STOPPING_ROUNDS = 30
LOCAL_CACHE_DIR = Path(".cache") / "autoresearch-tabular"
DEFAULT_OVERALL_BEST_DATASET_PATH = LOCAL_CACHE_DIR / "best_overall_engineered_dataset.parquet.gzip"

FIXED_XGB_PARAMS = {
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "scale_pos_weight": 1.0,
    "n_estimators": 100,
}

FEATURE_POLICY_BASE = {
    "screen_k": None,
    "feature_cap": None,
    "include_raw_numeric": True,
    "ratio_pairs": (),
    "multiply_pairs": (),
    "add_numeric_missing_flags": True,
}

STARTING_RATIO_PAIRS = (
    ("tot_mthly_obligation_accts_3m", "tot_pymt_amount_accts_3m"),
    ("tot_bal_bc_accts_3m", "tot_pymt_amount_bc_accts_3m"),
    ("tot_mthly_obligation_bc_accts_3m", "tot_pymt_amount_bc_accts_3m"),
    ("tot_bal_bc_accts_3m", "tot_mthly_obligation_bc_accts_3m"),
    ("tot_sched_mthly_pymt_open_mtg_trds_12m", "tot_sched_mthly_pymt_all_trds_12m"),
)

STARTING_MULTIPLY_PAIRS = (
    ("num_open_sat_inst_trds_24m_plus", "tot_sched_mthly_pymt_open_inst_trds_12m"),
)

BURDEN_RATIO_PAIRS = (
    ("monthly_debt", "stated_monthly_income"),
    ("tot_sched_mthly_pymt_all_trds_12m", "stated_monthly_income"),
    ("tot_sched_mthly_pymt_open_mtg_trds_12m", "stated_monthly_income"),
    ("tot_sched_mthly_pymt_open_inst_trds_12m", "stated_monthly_income"),
    ("revolving_balance", "stated_monthly_income"),
)

BURDEN_MULTIPLY_PAIRS = (
    ("util_open_cc_trds_12m", "avg_bal_all_cc_trds_0_12m"),
    ("num_deduped_inq_12m", "monthly_debt"),
)

HYBRID_RATIO_PAIRS = BURDEN_RATIO_PAIRS + (
    ("tot_sched_mthly_pymt_open_mtg_trds_12m", "tot_sched_mthly_pymt_all_trds_12m"),
)

HYBRID_MULTIPLY_PAIRS = (
    ("util_open_cc_trds_12m", "avg_bal_all_cc_trds_0_12m"),
)


FEATURE_POLICIES = [
    {
        **FEATURE_POLICY_BASE,
        "name": "starting_mtg_share_plus_inst_scale_mul",
        "ratio_pairs": STARTING_RATIO_PAIRS,
        "multiply_pairs": STARTING_MULTIPLY_PAIRS,
    },
    {
        **FEATURE_POLICY_BASE,
        "name": "burden_pairs_cap192_nomiss",
        "ratio_pairs": BURDEN_RATIO_PAIRS,
        "multiply_pairs": BURDEN_MULTIPLY_PAIRS,
        "feature_cap": 192,
        "add_numeric_missing_flags": False,
    },
]

BASELINE_NAME = "split_dataset_raw"


# Helpers

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    if denom <= 1e-12:
        return 0.0
    return float(abs((x @ y) / denom))


# Feature matrix helpers

def append_feature(
    parts: list[list[np.ndarray]],
    arrays: list[np.ndarray],
    feature_names: list[str],
    name: str,
) -> None:
    for frame_parts, array in zip(parts, arrays, strict=True):
        frame_parts.append(array.astype(np.float64, copy=False))
    feature_names.append(name)


def score_numeric_column(series: pd.Series, y_train: np.ndarray) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(values)
    score = 0.0
    if np.any(missing):
        score = max(score, safe_corr(missing.astype(np.float64), y_train))
    finite = np.isfinite(values)
    if not np.any(finite):
        return score
    median = float(np.nanmedian(values[finite]))
    filled = np.where(missing, median, values)
    return max(score, safe_corr(filled, y_train))


def apply_feature_cap(
    matrices: list[np.ndarray],
    y_train: np.ndarray,
    feature_names: list[str],
    feature_cap: int | None,
) -> tuple[list[np.ndarray], list[str], list[float]]:
    num_features = matrices[0].shape[1]
    scores = [safe_corr(matrices[0][:, idx], y_train) for idx in range(num_features)]
    if feature_cap is None or feature_cap <= 0 or num_features <= feature_cap:
        return matrices, feature_names, scores

    ranked_idx = sorted(range(num_features), key=lambda idx: (-scores[idx], feature_names[idx]))
    selected_idx = ranked_idx[:feature_cap]
    selected_idx.sort()
    capped_matrices = [matrix[:, selected_idx] for matrix in matrices]
    capped_names = [feature_names[idx] for idx in selected_idx]
    capped_scores = [scores[idx] for idx in selected_idx]
    return capped_matrices, capped_names, capped_scores


def build_numeric_features(
    train_frame: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    columns: list[str],
    feature_policy: dict,
    excluded_raw_columns: set[str],
    parts: list[list[np.ndarray]],
    feature_names: list[str],
) -> None:
    all_frames = [train_frame, *other_frames]
    for column in columns:
        series_list = [pd.to_numeric(frame[column], errors="coerce") for frame in all_frames]
        train_values = series_list[0].to_numpy(dtype=np.float64, na_value=np.nan)
        exclude_raw_column = column in excluded_raw_columns

        if feature_policy["add_numeric_missing_flags"] and not exclude_raw_column:
            missing_flags = [series.isna().to_numpy(dtype=np.float64) for series in series_list]
            if np.any(missing_flags[0]):
                append_feature(
                    parts,
                    missing_flags,
                    feature_names,
                    f"{column}__missing",
                )

        finite_train = train_values[np.isfinite(train_values)]
        fill_value = float(np.nanmedian(finite_train)) if len(finite_train) else 0.0
        arrays = [series.fillna(fill_value).to_numpy(dtype=np.float64) for series in series_list]
        if feature_policy["include_raw_numeric"] and not exclude_raw_column:
            append_feature(
                parts,
                arrays,
                feature_names,
                column,
            )


def numeric_combo_arrays(
    train_frame: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    left_column: str,
    right_column: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_frames = [train_frame, *other_frames]
    left_series_list = [pd.to_numeric(frame[left_column], errors="coerce") for frame in all_frames]
    right_series_list = [pd.to_numeric(frame[right_column], errors="coerce") for frame in all_frames]

    left_train = left_series_list[0].to_numpy(dtype=np.float64, na_value=np.nan)
    right_train = right_series_list[0].to_numpy(dtype=np.float64, na_value=np.nan)
    left_fill = float(np.nanmedian(left_train[np.isfinite(left_train)])) if np.isfinite(left_train).any() else 0.0
    right_fill = float(np.nanmedian(right_train[np.isfinite(right_train)])) if np.isfinite(right_train).any() else 0.0

    left_arrays = [series.fillna(left_fill).to_numpy(dtype=np.float64) for series in left_series_list]
    right_arrays = [series.fillna(right_fill).to_numpy(dtype=np.float64) for series in right_series_list]
    return left_arrays, right_arrays


def divide_feature_arrays(
    numerator_arrays: list[np.ndarray],
    denominator_arrays: list[np.ndarray],
    eps: float = 1e-6,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for numerator, denominator in zip(numerator_arrays, denominator_arrays, strict=True):
        safe_denominator = np.where(np.abs(denominator) < eps, np.nan, denominator)
        divided = np.divide(
            numerator,
            safe_denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=np.isfinite(safe_denominator),
        )
        outputs.append(np.nan_to_num(divided, nan=0.0, posinf=0.0, neginf=0.0))
    return outputs


# Model helpers

def fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[xgb.Booster, dict]:
    train_dmatrix = xgb.DMatrix(x_train, label=y_train)
    val_dmatrix = xgb.DMatrix(x_val, label=y_val)
    train_params = dict(FIXED_XGB_PARAMS)
    num_boost_round = int(train_params.pop("n_estimators"))

    evals_result: dict = {}
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": RANDOM_SEED,
            "max_delta_step": 1.0,
            **train_params,
        },
        dtrain=train_dmatrix,
        num_boost_round=num_boost_round,
        evals=[(train_dmatrix, "train"), (val_dmatrix, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
        evals_result=evals_result,
    )
    return booster, evals_result


def predict_scores(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    dmatrix = xgb.DMatrix(x)
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, best_iteration + 1))


# Feature engineering


def build_pair_features(
    train_frame: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    variable_combinations: list[tuple[str, str]],
    parts: list[list[np.ndarray]],
    feature_names: list[str],
    suffix: str,
    combine_arrays: Callable[[list[np.ndarray], list[np.ndarray]], list[np.ndarray]],
) -> None:
    for left_column, right_column in variable_combinations:
        left_arrays, right_arrays = numeric_combo_arrays(
            train_frame=train_frame,
            other_frames=other_frames,
            left_column=left_column,
            right_column=right_column,
        )
        pair_arrays = combine_arrays(left_arrays, right_arrays)
        append_feature(
            parts,
            pair_arrays,
            feature_names,
            f"{left_column}__{suffix}__{right_column}",
        )


def engineer_feature_views(dataset: dict, feature_policy: dict) -> dict:
    train_frame = dataset["frame_train"]
    other_frames = [dataset["frame_val"], dataset["frame_test"], dataset["frame_oot"]]
    frames = [train_frame, *other_frames]
    numeric_columns = list(dataset["column_types"]["numeric"])
    raw_scores = {
        column: score_numeric_column(train_frame[column], dataset["y_train"])
        for column in numeric_columns
    }
    screen_k = feature_policy["screen_k"]
    if screen_k is not None and screen_k > 0 and screen_k < len(numeric_columns):
        ranked = sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))
        numeric_columns = [column for column, _ in ranked[:screen_k]]

    numeric_column_set = set(numeric_columns)
    ratio_pairs = [
        (left_column, right_column)
        for left_column, right_column in feature_policy["ratio_pairs"]
        if left_column in numeric_column_set and right_column in numeric_column_set
    ]
    multiply_pairs = [
        (left_column, right_column)
        for left_column, right_column in feature_policy["multiply_pairs"]
        if left_column in numeric_column_set and right_column in numeric_column_set
    ]
    paired_columns = {
        column
        for left_column, right_column in [*ratio_pairs, *multiply_pairs]
        for column in (left_column, right_column)
    }

    parts: list[list[np.ndarray]] = [[] for _ in frames]
    feature_names: list[str] = []

    build_numeric_features(
        train_frame=train_frame,
        other_frames=other_frames,
        columns=numeric_columns,
        feature_policy=feature_policy,
        excluded_raw_columns=paired_columns,
        parts=parts,
        feature_names=feature_names,
    )

    build_pair_features(
        train_frame=train_frame,
        other_frames=other_frames,
        variable_combinations=ratio_pairs,
        parts=parts,
        feature_names=feature_names,
        suffix="ratio",
        combine_arrays=divide_feature_arrays,
    )
    build_pair_features(
        train_frame=train_frame,
        other_frames=other_frames,
        variable_combinations=multiply_pairs,
        parts=parts,
        feature_names=feature_names,
        suffix="mul",
        combine_arrays=lambda left_arrays, right_arrays: [
            (left * right).astype(np.float64, copy=False)
            for left, right in zip(left_arrays, right_arrays, strict=True)
        ],
    )

    matrices = []
    for frame, frame_parts in zip(frames, parts, strict=True):
        if frame_parts:
            matrices.append(np.column_stack(frame_parts).astype(np.float64, copy=False))
        else:
            matrices.append(np.empty((len(frame), 0), dtype=np.float64))

    matrices, feature_names, engineered_scores = apply_feature_cap(
        matrices=matrices,
        y_train=dataset["y_train"],
        feature_names=feature_names,
        feature_cap=feature_policy["feature_cap"],
    )

    return {
        "x_train": matrices[0],
        "x_val": matrices[1],
        "x_test": matrices[2],
        "x_oot": matrices[3],
        "feature_names": feature_names,
        "feature_counts": {
            "numeric": len(feature_names),
            "bool": 0,
            "categorical": 0,
            "datetime": 0,
        },
        "effective_column_types": {
            "numeric": list(numeric_columns),
            "bool": [],
            "categorical": [],
            "datetime": [],
        },
        "raw_column_scores": raw_scores,
        "engineered_feature_scores": engineered_scores,
    }


def engineer_baseline_views(dataset: dict) -> dict:
    numeric_columns = list(dataset["column_types"]["numeric"])
    bool_columns = list(dataset["column_types"]["bool"])
    feature_names = [*numeric_columns, *bool_columns]
    matrices = [
        np.column_stack(
            [
                pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                for column in feature_names
            ]
        ).astype(np.float64, copy=False)
        for frame in [
            dataset["frame_train"],
            dataset["frame_val"],
            dataset["frame_test"],
            dataset["frame_oot"],
        ]
    ]

    return {
        "x_train": matrices[0],
        "x_val": matrices[1],
        "x_test": matrices[2],
        "x_oot": matrices[3],
        "feature_names": feature_names,
        "feature_counts": {
            "numeric": len(numeric_columns),
            "bool": len(bool_columns),
            "categorical": 0,
            "datetime": 0,
        },
        "effective_column_types": {
            "numeric": numeric_columns,
            "bool": bool_columns,
            "categorical": [],
            "datetime": [],
        },
        "raw_column_scores": {},
        "engineered_feature_scores": [
            safe_corr(matrices[0][:, idx], dataset["y_train"])
            for idx in range(matrices[0].shape[1])
        ],
    }


# Evaluation

def describe_feature_policy(feature_policy: dict) -> str:
    return (
        f"fe={feature_policy['name']} "
        f"screen_k={feature_policy['screen_k']} "
        f"feature_cap={feature_policy['feature_cap']} "
        f"raw_numeric={int(feature_policy['include_raw_numeric'])} "
        f"ratio_pairs={len(feature_policy['ratio_pairs'])} "
        f"multiply_pairs={len(feature_policy['multiply_pairs'])} "
        f"missing_flags=num:{int(feature_policy['add_numeric_missing_flags'])}"
    )


def evaluate_run(dataset: dict, trial: int, feature_policy: dict | None = None) -> dict:
    started = time.time()
    if feature_policy is None:
        views = engineer_baseline_views(dataset=dataset)
        policy_label = "baseline=split_dataset_raw numeric+bool_only no_feature_policy"
        description = "single fixed xgboost on split dataset raw numeric and bool columns only"
        config = {
            "name": BASELINE_NAME,
            "baseline": True,
            "xgboost_params": FIXED_XGB_PARAMS,
        }
    else:
        views = engineer_feature_views(dataset=dataset, feature_policy=feature_policy)
        policy_label = describe_feature_policy(feature_policy)
        description = f"single fixed xgboost with {feature_policy['name']} feature engineering"
        config = {
            "name": feature_policy["name"],
            "feature_policy": feature_policy,
            "xgboost_params": FIXED_XGB_PARAMS,
        }

    booster, evals_result = fit_xgboost(
        x_train=views["x_train"],
        y_train=dataset["y_train"],
        x_val=views["x_val"],
        y_val=dataset["y_val"],
    )
    val_scores = predict_scores(booster, views["x_val"])
    test_scores = predict_scores(booster, views["x_test"])
    oot_scores = predict_scores(booster, views["x_oot"])
    val_auc = auc_score(dataset["y_val"], val_scores)
    test_auc = auc_score(dataset["y_test"], test_scores)
    oot_auc = auc_score(dataset["y_oot"], oot_scores)
    pos_rate = float(np.mean(dataset["y_train"]))

    result = {
        "trial": trial,
        "val_auc": val_auc,
        "initial_val_auc": val_auc,
        "test_auc": test_auc,
        "oot_auc": oot_auc,
        "num_features": len(views["feature_names"]),
        "feature_names": views["feature_names"],
        "feature_counts": views["feature_counts"],
        "raw_column_scores": views["raw_column_scores"],
        "engineered_feature_scores": views["engineered_feature_scores"],
        "policy": policy_label,
        "description": description,
        "config": config,
        "best_iteration": int(booster.best_iteration),
        "train_auc_history_tail": evals_result["train"]["auc"][-5:],
        "val_auc_history_tail": evals_result["val"]["auc"][-5:],
        "class_balance": f"{pos_rate * 100.0:.2f}%/{(1.0 - pos_rate) * 100.0:.2f}%",
        "split_source": dataset["split_source"],
        "time_column": dataset["time_column"],
        "effective_column_types": views["effective_column_types"],
        "elapsed_seconds": time.time() - started,
    }
    print(
        f"trial={trial:02d} val_auc={val_auc:.6f} test_auc={test_auc:.6f} "
        f"oot_auc={oot_auc:.6f} features={len(views['feature_names']):04d} "
        f"policy={config['name']}"
    )
    return result


def maybe_save_overall_best_dataset(
    dataset: dict,
    best_result: dict,
    output_path: Path,
) -> bool:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    existing_val_auc = float("-inf")
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        existing_val_auc = float(existing.get("val_auc", float("-inf")))

    current_val_auc = float(best_result["val_auc"])
    if current_val_auc <= existing_val_auc:
        return False

    feature_policy = best_result["config"].get("feature_policy")
    views = (
        engineer_baseline_views(dataset=dataset)
        if feature_policy is None
        else engineer_feature_views(dataset=dataset, feature_policy=feature_policy)
    )
    engineered = pd.concat(
        [
            pd.DataFrame(views["x_train"], columns=views["feature_names"]).assign(
                target=dataset["y_train"].astype(np.int64),
                split="train",
            ),
            pd.DataFrame(views["x_val"], columns=views["feature_names"]).assign(
                target=dataset["y_val"].astype(np.int64),
                split="val",
            ),
            pd.DataFrame(views["x_test"], columns=views["feature_names"]).assign(
                target=dataset["y_test"].astype(np.int64),
                split="test",
            ),
            pd.DataFrame(views["x_oot"], columns=views["feature_names"]).assign(
                target=dataset["y_oot"].astype(np.int64),
                split="oot",
            ),
        ],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(output_path, index=False)

    metadata = {
        "num_features": len(views["feature_names"]),
        "feature_names": views["feature_names"],
        "feature_counts": views["feature_counts"],
        "split_source": dataset["split_source"],
        "time_column": dataset["time_column"],
    }
    if feature_policy is None:
        metadata["baseline"] = True
    else:
        metadata["feature_policy"] = feature_policy["name"]
    metadata.update(
        {
            "val_auc": current_val_auc,
            "test_auc": float(best_result["test_auc"]),
            "oot_auc": float(best_result["oot_auc"]),
            "policy": best_result["policy"],
            "description": best_result["description"],
            "class_balance": best_result["class_balance"],
            "best_iteration": int(best_result["best_iteration"]),
            "saved_from_trial": int(best_result["trial"]),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def run_baseline(cache_dir: Path) -> tuple[dict, dict]:
    dataset = load_dataset(cache_dir=cache_dir)
    return evaluate_run(dataset=dataset, trial=1), dataset


def run_search(cache_dir: Path) -> tuple[dict, dict]:
    dataset = load_dataset(cache_dir=cache_dir)
    start = time.time()
    best = None

    for trial, feature_policy in enumerate(FEATURE_POLICIES, start=1):
        if time.time() - start > max(TIME_BUDGET, 600.0):
            break
        result = evaluate_run(dataset=dataset, trial=trial, feature_policy=feature_policy)
        if best is None or result["val_auc"] > best["val_auc"]:
            best = result

    assert best is not None
    best["elapsed_seconds"] = time.time() - start
    return best, dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run only the raw split-dataset baseline without applying FEATURE_POLICIES.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=LOCAL_CACHE_DIR,
        help="Location of the prepared dataset cache.",
    )
    parser.add_argument(
        "--save-overall-best-dataset",
        type=Path,
        default=DEFAULT_OVERALL_BEST_DATASET_PATH,
        help="Parquet path for the best overall engineered dataset across runs; overwritten only when val_auc improves.",
    )
    args = parser.parse_args()

    best, dataset = (
        run_baseline(cache_dir=args.cache_dir)
        if args.baseline
        else run_search(cache_dir=args.cache_dir)
    )

    if args.save_overall_best_dataset is not None:
        updated = maybe_save_overall_best_dataset(
            dataset=dataset,
            best_result=best,
            output_path=args.save_overall_best_dataset,
        )
        print(f"saved_overall_best_dataset: {args.save_overall_best_dataset} updated={int(updated)}")

    print(json.dumps(best, indent=2))
    print(f"val_auc: {best['val_auc']:.6f}")


if __name__ == "__main__":
    main()
