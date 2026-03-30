import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from prepare import TIME_BUDGET, auc_score, default_cache_dir, load_dataset

RANDOM_SEED = 1337
MAX_TRIALS = 120
SEARCH_TIME_BUDGET = max(TIME_BUDGET, 600.0)
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 30
LOCAL_CACHE_DIR = default_cache_dir()

BASELINE_PARAMS = {
    "eta": 0.10,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_estimators": 400,
}

FEATURE_CAPS = [136]
HYPERPARAM_GRID = [
    {
        "eta": 0.095,
        "max_depth": 3,
        "min_child_weight": 7.0,
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "n_estimators": 550,
    },
]
SAMPLING_PLANS = [
    {"name": "none", "target_pos_rate": None, "scale_pos_weight_mode": "none"},
    {"name": "undersample_14", "target_pos_rate": 0.14, "scale_pos_weight_mode": "none"},
    {"name": "oversample_14", "target_pos_rate": 0.14, "scale_pos_weight_mode": "none"},
]

BASELINE_SAMPLING = {"name": "none", "target_pos_rate": None, "scale_pos_weight_mode": "none"}


# Sampling helpers

def balanced_weight(y: np.ndarray) -> float:
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    return negatives / positives


def resample_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    target_pos_rate: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "none" or target_pos_rate is None:
        return x_train, y_train

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x_train, y_train

    if method.startswith("undersample"):
        target_neg = int(round(len(pos_idx) * (1.0 - target_pos_rate) / target_pos_rate))
        target_neg = min(len(neg_idx), max(target_neg, len(pos_idx)))
        chosen_neg = rng.choice(neg_idx, size=target_neg, replace=False)
        chosen = rng.permutation(np.concatenate([pos_idx, chosen_neg]))
        return x_train[chosen], y_train[chosen]
    if method.startswith("oversample"):
        target_pos = int(round(len(neg_idx) * target_pos_rate / (1.0 - target_pos_rate)))
        target_pos = max(target_pos, len(pos_idx))
        extra_pos = rng.choice(pos_idx, size=target_pos - len(pos_idx), replace=True)
        chosen = rng.permutation(np.concatenate([neg_idx, pos_idx, extra_pos]))
        return x_train[chosen], y_train[chosen]
    raise ValueError(f"Unknown sampling method: {method}")


def effective_scale_pos_weight(mode: str, y_train: np.ndarray) -> float:
    if mode == "none":
        return 1.0
    if mode == "sqrt_balanced":
        return max(balanced_weight(y_train) ** 0.5, 1.0)
    raise ValueError(f"Unknown scale_pos_weight mode: {mode}")


# Model helpers

def fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    scale_pos_weight: float = 1.0,
    max_delta_step: float | None = 1.0,
) -> tuple[xgb.Booster, dict]:
    train_params = dict(params)
    num_boost_round = int(train_params.pop("n_estimators", NUM_BOOST_ROUND))
    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
        **train_params,
    }
    if max_delta_step is not None:
        booster_params["max_delta_step"] = max_delta_step

    evals_result: dict = {}
    booster = xgb.train(
        params=booster_params,
        dtrain=xgb.DMatrix(x_train, label=y_train),
        num_boost_round=num_boost_round,
        evals=[
            (xgb.DMatrix(x_train, label=y_train), "train"),
            (xgb.DMatrix(x_val, label=y_val), "val"),
        ],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
        evals_result=evals_result,
    )
    return booster, evals_result


def predict_scores(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        return booster.predict(xgb.DMatrix(x))
    return booster.predict(xgb.DMatrix(x), iteration_range=(0, best_iteration + 1))


def select_top_features(
    booster: xgb.Booster,
    num_features: int,
    feature_cap: int | None,
) -> tuple[list[int], list[float]]:
    gain_scores = booster.get_score(importance_type="gain")
    importance = [float(gain_scores.get(f"f{idx}", 0.0)) for idx in range(num_features)]
    if feature_cap is None or feature_cap >= num_features:
        return list(range(num_features)), importance

    ranked_idx = sorted(range(num_features), key=lambda idx: (-importance[idx], idx))
    selected_idx = ranked_idx[:feature_cap]
    return selected_idx, [importance[idx] for idx in selected_idx]


# Search helpers

def config_seed(config: dict) -> int:
    payload = json.dumps(
        {
            "feature_cap": config["feature_cap"],
            "sampling": config["sampling"],
            "hyperparams": config["hyperparams"],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return RANDOM_SEED + int(digest[:8], 16)


def candidate_configs() -> list[dict]:
    return [
        {
            "name": "search",
            "feature_cap": feature_cap,
            "sampling": sampling,
            "hyperparams": hyperparams,
        }
        for feature_cap in FEATURE_CAPS
        for sampling in SAMPLING_PLANS
        for hyperparams in HYPERPARAM_GRID
    ]

def describe_policy(config: dict, num_features: int, scale_pos_weight: float) -> str:
    hyperparams = config["hyperparams"]
    sampling = config["sampling"]
    target_pos_rate = sampling["target_pos_rate"]
    target_pos_rate_str = "none" if target_pos_rate is None else f"{target_pos_rate:.2f}"
    feature_cap = "all" if config["feature_cap"] is None else str(config["feature_cap"])
    return (
        f"name={config['name']} "
        f"feature_cap={feature_cap} "
        f"num_features={num_features} "
        f"sampling={sampling['name']} "
        f"target_pos_rate={target_pos_rate_str} "
        f"weight={sampling['scale_pos_weight_mode']} "
        f"scale_pos_weight={scale_pos_weight:.3f} "
        f"eta={hyperparams['eta']:.3f} "
        f"max_depth={hyperparams['max_depth']} "
        f"min_child_weight={hyperparams['min_child_weight']:.1f} "
        f"subsample={hyperparams['subsample']:.2f} "
        f"colsample_bytree={hyperparams['colsample_bytree']:.2f} "
        f"reg_lambda={hyperparams['reg_lambda']:.1f} "
        f"reg_alpha={hyperparams['reg_alpha']:.1f} "
        f"n_estimators={int(hyperparams.get('n_estimators', NUM_BOOST_ROUND))}"
    )


def summarize_policy(config: dict, num_features: int) -> str:
    hyperparams = config["hyperparams"]
    sampling = config["sampling"]["name"]
    depth = "a shallow tree" if hyperparams["max_depth"] <= 3 else f"depth {hyperparams['max_depth']}"
    return (
        f"try {sampling} keep {num_features} features "
        f"with {depth} eta {hyperparams['eta']:.3f}, "
        f"min child weight {hyperparams['min_child_weight']:.1f}"
    )


def print_trial(result: dict) -> None:
    hyperparams = result["config"]["hyperparams"]
    sampling = result["config"]["sampling"]
    feature_cap = "all" if result["config"]["feature_cap"] is None else str(result["config"]["feature_cap"])
    print(
        f"trial={result['trial']:02d} val_auc={result['val_auc']:.6f} "
        f"initial_val_auc={result['initial_val_auc']:.6f} "
        f"test_auc={result['test_auc']:.6f} oot_auc={result['oot_auc']:.6f} "
        f"features={result['num_features']:03d} "
        f"cap={feature_cap} "
        f"sampling={sampling['name']} weight={sampling['scale_pos_weight_mode']} "
        f"depth={hyperparams['max_depth']} eta={hyperparams['eta']:.3f} "
        f"mcw={hyperparams['min_child_weight']:.1f}"
    )

# Execution

def run_trial(
    data: dict,
    trial: int,
    config: dict,
    original_class_balance: str,
) -> tuple[dict, tuple]:
    sampled_x, sampled_y = resample_training_data(
        x_train=data["x_train"],
        y_train=data["y_train"],
        method=config["sampling"]["name"],
        target_pos_rate=config["sampling"]["target_pos_rate"],
        seed=config_seed(config),
    )
    scale_pos_weight = effective_scale_pos_weight(config["sampling"]["scale_pos_weight_mode"], sampled_y)
    is_baseline = config["name"] == "baseline"

    initial_booster, initial_evals = fit_xgboost(
        x_train=sampled_x,
        y_train=sampled_y,
        x_val=data["x_val"],
        y_val=data["y_val"],
        params=config["hyperparams"],
        scale_pos_weight=scale_pos_weight,
        max_delta_step=None if is_baseline else 1.0,
    )

    selected_idx, feature_importance = select_top_features(
        booster=initial_booster,
        num_features=data["x_train"].shape[1],
        feature_cap=config["feature_cap"],
    )
    signature = (
        tuple(selected_idx),
        config["sampling"]["name"],
        config["sampling"]["scale_pos_weight_mode"],
        tuple(sorted(config["hyperparams"].items())),
    )

    selected_names = [data["feature_names"][idx] for idx in selected_idx]
    train_view = data["x_train"][:, selected_idx]
    val_view = data["x_val"][:, selected_idx]
    test_view = data["x_test"][:, selected_idx]
    oot_view = data["x_oot"][:, selected_idx]
    sampled_view = sampled_x[:, selected_idx]

    booster = initial_booster
    evals_result = initial_evals
    if config["feature_cap"] is not None:
        booster, evals_result = fit_xgboost(
            x_train=sampled_view,
            y_train=sampled_y,
            x_val=val_view,
            y_val=data["y_val"],
            params=config["hyperparams"],
            scale_pos_weight=scale_pos_weight,
        )

    val_auc = auc_score(data["y_val"], predict_scores(booster, val_view))
    test_auc = auc_score(data["y_test"], predict_scores(booster, test_view))
    oot_auc = auc_score(data["y_oot"], predict_scores(booster, oot_view))
    initial_val_auc = auc_score(data["y_val"], predict_scores(initial_booster, data["x_val"]))
    policy = describe_policy(config, len(selected_names), scale_pos_weight)
    description = (
        "plain xgboost baseline without tuning or feature pruning"
        if is_baseline
        else summarize_policy(config, len(selected_names))
    )
    pos_rate = float(np.mean(sampled_y))
    sampled_class_balance = f"{pos_rate * 100.0:.2f}%/{(1.0 - pos_rate) * 100.0:.2f}%"

    result = {
        "trial": trial,
        "val_auc": val_auc,
        "initial_val_auc": initial_val_auc,
        "test_auc": test_auc,
        "oot_auc": oot_auc,
        "num_features": len(selected_names),
        "feature_names": selected_names,
        "feature_importance_gain": feature_importance,
        "policy": policy,
        "description": description,
        "config": {
            "name": config["name"],
            "feature_cap": config["feature_cap"],
            "sampling": config["sampling"],
            "scale_pos_weight": scale_pos_weight,
            "hyperparams": config["hyperparams"],
        },
        "initial_best_iteration": int(initial_booster.best_iteration),
        "best_iteration": int(booster.best_iteration),
        "train_auc_history_tail": evals_result["train"]["auc"][-5:],
        "val_auc_history_tail": evals_result["val"]["auc"][-5:],
        "class_balance": sampled_class_balance,
        "original_class_balance": original_class_balance,
        "sampled_class_balance": sampled_class_balance,
    }
    return result, signature


def run_baseline(data: dict) -> dict:
    pos_rate = float(np.mean(data["y_train"]))
    original_class_balance = f"{pos_rate * 100.0:.2f}%/{(1.0 - pos_rate) * 100.0:.2f}%"
    config = {
        "name": "baseline",
        "feature_cap": None,
        "sampling": BASELINE_SAMPLING,
        "hyperparams": BASELINE_PARAMS,
    }
    result, _ = run_trial(data=data, trial=1, config=config, original_class_balance=original_class_balance)
    result["elapsed_seconds"] = 0.0
    print_trial(result)
    return result


def run_search(data: dict) -> dict:
    pos_rate = float(np.mean(data["y_train"]))
    original_class_balance = f"{pos_rate * 100.0:.2f}%/{(1.0 - pos_rate) * 100.0:.2f}%"
    start = time.time()
    best = None
    evaluated = 0
    seen_signatures = set()

    for config in candidate_configs():
        if evaluated >= MAX_TRIALS or time.time() - start > SEARCH_TIME_BUDGET:
            break

        trial_number = evaluated + 1
        result, signature = run_trial(
            data=data,
            trial=trial_number,
            config=config,
            original_class_balance=original_class_balance,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        evaluated += 1
        result["trial"] = evaluated
        print_trial(result)
        if best is None or result["val_auc"] > best["val_auc"]:
            best = result

    assert best is not None
    best["elapsed_seconds"] = time.time() - start
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the required untuned baseline instead of the current search policy.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=LOCAL_CACHE_DIR,
        help="Location of the prepared dataset cache.",
    )
    args = parser.parse_args()

    dataset = load_dataset(cache_dir=args.cache_dir)
    data = {
        "x_train": dataset["frame_train"].to_numpy(dtype=np.float64, copy=False),
        "y_train": dataset["y_train"],
        "x_val": dataset["frame_val"].to_numpy(dtype=np.float64, copy=False),
        "y_val": dataset["y_val"],
        "x_test": dataset["frame_test"].to_numpy(dtype=np.float64, copy=False),
        "y_test": dataset["y_test"],
        "x_oot": dataset["frame_oot"].to_numpy(dtype=np.float64, copy=False),
        "y_oot": dataset["y_oot"],
        "feature_names": dataset["feature_names"],
    }
    best = run_baseline(data=data) if args.baseline else run_search(data=data)
    print(json.dumps(best, indent=2))
    print(f"val_auc: {best['val_auc']:.6f}")


if __name__ == "__main__":
    main()
