# train.py

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import AppConfig
from extract_db import JonathonMetadataExtractor
from extract_logs_table import ErrorLogTableExtractor
from features import build_training_features, data_quality_checks
from storage_utils import GCSOutputStore

logger = logging.getLogger(__name__)


class EarlyExitAfterDiagnosticsLogs(Exception):
    pass


@dataclass
class TrainOutputs:
    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path
    metrics_csv_path: Path
    feature_importance_path: Path
    feature_frame_path: Path


def _split_time_based(frame: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values(by=["logical_date"], kind="mergesort").reset_index(drop=True)
    max_date = sorted_frame["logical_date"].max()
    split_ts = max_date - pd.Timedelta(days=test_days)

    train = sorted_frame[sorted_frame["logical_date"] < split_ts].copy()
    test = sorted_frame[sorted_frame["logical_date"] >= split_ts].copy()

    total_rows = len(sorted_frame)
    min_train_rows = max(100, int(total_rows * 0.2))

    train_unique_labels = sorted(train["label_failed"].dropna().unique().tolist()) if "label_failed" in train.columns else []

    fallback_needed = (
        train.empty
        or test.empty
        or len(train) < min_train_rows
        or len(train_unique_labels) < 2
    )

    if fallback_needed:
        logger.warning(
            "Time-based split fallback triggered. train_rows=%s test_rows=%s min_train_rows=%s train_label_classes=%s test_days=%s",
            len(train),
            len(test),
            min_train_rows,
            len(train_unique_labels),
            test_days,
        )
        split_idx = int(total_rows * 0.8)
        train = sorted_frame.iloc[:split_idx].copy()
        test = sorted_frame.iloc[split_idx:].copy()

    train_min = train["logical_date"].min() if "logical_date" in train.columns and not train.empty else pd.NaT
    train_max = train["logical_date"].max() if "logical_date" in train.columns and not train.empty else pd.NaT
    test_min = test["logical_date"].min() if "logical_date" in test.columns and not test.empty else pd.NaT
    test_max = test["logical_date"].max() if "logical_date" in test.columns and not test.empty else pd.NaT

    logger.info(
        "Split summary: split_ts=%s train_rows=%s test_rows=%s train_range=[%s, %s] test_range=[%s, %s]",
        split_ts,
        len(train),
        len(test),
        train_min,
        train_max,
        test_min,
        test_max,
    )
    return cast(pd.DataFrame, train), cast(pd.DataFrame, test)


def _split_time_based_three_way(
    frame: pd.DataFrame,
    val_days: int,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values(by=["logical_date"], kind="mergesort").reset_index(drop=True)
    max_date = sorted_frame["logical_date"].max()

    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = sorted_frame[sorted_frame["logical_date"] < val_start].copy()
    test = sorted_frame[sorted_frame["logical_date"] >= test_start].copy()
    val = sorted_frame[
        (sorted_frame["logical_date"] >= val_start) & (sorted_frame["logical_date"] < test_start)
    ].copy()

    total_rows = len(sorted_frame)
    min_train_rows = max(100, int(total_rows * 0.5))
    min_val_rows = max(30, int(total_rows * 0.1))
    min_test_rows = max(30, int(total_rows * 0.1))

    train_unique_labels = sorted(train["label_failed"].dropna().unique().tolist()) if "label_failed" in train.columns else []
    val_unique_labels = sorted(val["label_failed"].dropna().unique().tolist()) if "label_failed" in val.columns else []

    fallback_needed = (
        train.empty
        or val.empty
        or test.empty
        or len(train) < min_train_rows
        or len(val) < min_val_rows
        or len(test) < min_test_rows
        or len(train_unique_labels) < 2
        or len(val_unique_labels) < 2
    )

    if fallback_needed:
        logger.warning(
            "Three-way split fallback triggered. train_rows=%s val_rows=%s test_rows=%s val_days=%s test_days=%s",
            len(train),
            len(val),
            len(test),
            val_days,
            test_days,
        )
        split_test = int(total_rows * 0.8)
        split_val = int(split_test * 0.8)
        train = sorted_frame.iloc[:split_val].copy()
        val = sorted_frame.iloc[split_val:split_test].copy()
        test = sorted_frame.iloc[split_test:].copy()

    logger.info(
        "Three-way split summary: train_rows=%d val_rows=%d test_rows=%d",
        len(train),
        len(val),
        len(test),
    )
    return cast(pd.DataFrame, train), cast(pd.DataFrame, val), cast(pd.DataFrame, test)


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [
        col
        for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("string") or str(X[col].dtype) == "category"
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def _compute_best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true).reshape(-1)
    probs_arr = np.asarray(probs, dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_true_arr) & np.isfinite(probs_arr)

    if valid_mask.sum() == 0:
        logger.warning("Threshold selection received no finite rows; using default threshold=0.5")
        return 0.5

    y_true_valid = y_true_arr[valid_mask].astype(int)
    probs_valid = np.clip(probs_arr[valid_mask], 0.0, 1.0)

    if len(np.unique(y_true_valid)) < 2:
        logger.warning("Threshold selection has a single class in validation labels; using default threshold=0.5")
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true_valid, probs_valid)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_scores))])


def _evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).reshape(-1)
    probs_arr = np.asarray(probs, dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_true_arr) & np.isfinite(probs_arr)

    if valid_mask.sum() == 0:
        cm = np.array([[0, 0], [0, 0]], dtype=int)
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "threshold": float(threshold),
            "precision": float("nan"),
            "recall": float("nan"),
            "confusion_matrix": cm.tolist(),
        }

    y_true_valid = y_true_arr[valid_mask].astype(int)
    probs_valid = np.clip(probs_arr[valid_mask], 0.0, 1.0)
    preds = (probs_valid > threshold).astype(int)
    cm = confusion_matrix(y_true_valid, preds, labels=[0, 1])

    return {
        "pr_auc": float(average_precision_score(y_true_valid, probs_valid)) if len(np.unique(y_true_valid)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true_valid, probs_valid)) if len(np.unique(y_true_valid)) > 1 else float("nan"),
        "threshold": float(threshold),
        "precision": float(cm[1, 1] / max(cm[:, 1].sum(), 1)),
        "recall": float(cm[1, 1] / max(cm[1, :].sum(), 1)),
        "confusion_matrix": cm.tolist(),
    }


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _xgb_param_grid(config: AppConfig) -> list[dict[str, float | int]]:
    base_n = config.model.xgb_n_estimators
    base_depth = config.model.xgb_max_depth
    base_lr = config.model.xgb_learning_rate
    base_subsample = config.model.xgb_subsample
    base_colsample = config.model.xgb_colsample_bytree

    n_values = sorted({max(100, base_n - 100), base_n, base_n + 100})
    depth_values = sorted({max(3, base_depth - 1), base_depth, base_depth + 1})
    lr_values = sorted({max(0.02, round(base_lr * 0.7, 4)), base_lr})
    subsample_values = sorted({max(0.6, round(base_subsample - 0.1, 2)), min(1.0, base_subsample)})
    colsample_values = sorted({max(0.6, round(base_colsample - 0.1, 2)), min(1.0, base_colsample)})

    grid: list[dict[str, float | int]] = []
    for depth in depth_values:
        for learning_rate in lr_values:
            for subsample in subsample_values:
                for colsample_bytree in colsample_values:
                    for n_estimators in n_values:
                        grid.append({
                            "n_estimators": n_estimators,
                            "max_depth": depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                        })
    return grid[:18]


def train_pipeline(config: AppConfig, start_date: date | None, end_date: date | None) -> TrainOutputs:
    config.ensure_dirs()

    extractor = JonathonMetadataExtractor(config)
    extracted = extractor.fetch(start_date=start_date, end_date=end_date)

    run_log_features = pd.DataFrame()
    if config.logs.enabled and config.logs.error_log_table:
        try:
            run_log_features = ErrorLogTableExtractor(config.logs).extract_for_runs(
                extracted.dag_run,
                extracted.task_instance,
            )
        except Exception as exc:
            logger.warning("Log extraction failed, continuing without log features: %s", exc)

    artifacts = build_training_features(
        dag_run_df=extracted.dag_run,
        task_instance_df=extracted.task_instance,
        dag_df=extracted.dag,
        run_log_features_df=run_log_features,
        cfg=config.features,
    )

    feature_frame = artifacts.feature_frame

    if feature_frame.empty:
        raise RuntimeError("Feature frame is empty; cannot train model.")

    quality = data_quality_checks(feature_frame)
    logger.info("Data quality summary: %s", quality)

    train_df, val_df, test_df = _split_time_based_three_way(
        feature_frame,
        val_days=config.model.val_days,
        test_days=config.model.test_days,
    )

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Insufficient data after split for train/val/test.")

    feature_cols = [
        col
        for col in feature_frame.columns
        if col not in {"label_failed", "state", "logical_date", "dag_id", "run_id"}
    ]

    X_train = cast(pd.DataFrame, train_df[feature_cols].copy())
    y_train = train_df["label_failed"].astype(int).to_numpy()
    X_val = cast(pd.DataFrame, val_df[feature_cols].copy())
    y_val = val_df["label_failed"].astype(int).to_numpy()
    X_test = cast(pd.DataFrame, test_df[feature_cols].copy())
    y_test = test_df["label_failed"].astype(int).to_numpy()

    # Drop entirely-NaN feature columns
    all_nan_cols = [col for col in feature_cols if X_train[col].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Dropping %d feature column(s) that are entirely NaN in training set: %s",
            len(all_nan_cols),
            all_nan_cols,
        )
        feature_cols = [col for col in feature_cols if col not in all_nan_cols]
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
    else:
        logger.info("No entirely-NaN feature columns found.")

    # Pre-training diagnostics
    logger.info("-" * 70)
    logger.info("TRAIN SIZE: %d rows x %d features", len(X_train), len(feature_cols))
    logger.info("VAL SIZE: %d rows x %d features", len(X_val), len(feature_cols))
    logger.info("TEST SIZE: %d rows x %d features", len(X_test), len(feature_cols))
    logger.info("LABEL dist (train) 0: %d  1: %d", int((y_train == 0).sum()), int((y_train == 1).sum()))
    logger.info("LABEL dist (val)   0: %d  1: %d", int((y_val == 0).sum()), int((y_val == 1).sum()))
    logger.info("LABEL dist (test)  0: %d  1: %d", int((y_test == 0).sum()), int((y_test == 1).sum()))
    logger.info("FEATURE COLUMNS (%d): %s", len(feature_cols), feature_cols)
    logger.info("FEATURE SAMPLE (first 5 rows):\n%s", X_train.head().to_string())
    logger.info("-" * 70)

    if config.model.exit_after_diagnostics_logs:
        logger.info(
            "EXIT_AFTER_DIAGNOSTICS_LOGS is enabled. Stopping after diagnostics log checkpoint."
        )
        raise EarlyExitAfterDiagnosticsLogs(
            "Intentional stop after diagnostics logs for validation."
        )

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    class_counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = float(class_counts[0] / max(class_counts[1], 1)) if len(class_counts) > 1 else 1.0

    try:
        model_type = "xgboost"
        from xgboost import XGBClassifier

        best_score = float("-inf")
        best_params: dict[str, float | int] = {}
        model = None
        early_stopping_rounds = 20

        for candidate_params in _xgb_param_grid(config):
            candidate = XGBClassifier(
                n_estimators=int(candidate_params["n_estimators"]),
                learning_rate=float(candidate_params["learning_rate"]),
                max_depth=int(candidate_params["max_depth"]),
                colsample_bytree=float(candidate_params["colsample_bytree"]),
                subsample=float(candidate_params["subsample"]),
                objective="binary:logistic",
                eval_metric="logloss",
                early_stopping_rounds=early_stopping_rounds,
                random_state=config.model.random_state,
                n_jobs=4,
                scale_pos_weight=scale_pos_weight,
            )
            candidate.fit(
                X_train_t,
                y_train,
                eval_set=[(X_val_t, y_val)],
                verbose=False,
            )

            candidate_val_probs = candidate.predict_proba(X_val_t)[:, 1]
            candidate_score = float(average_precision_score(y_val, candidate_val_probs)) if len(np.unique(y_val)) > 1 else float("-inf")

            logger.info(
                "Grid candidate params=%s best_iteration=%s val_pr_auc=%.6f",
                candidate_params,
                getattr(candidate, "best_iteration", "n/a"),
                candidate_score,
            )

            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params
                model = candidate

        if model is None:
            raise RuntimeError("XGBoost grid search did not produce a valid model.")

        print("\n" + "-" * 70)
        logger.info("Best XGBoost validation PR-AUC=%.6f params=%s", best_score, best_params)
        print("-" * 70)
        print("  GRID SEARCH RESULT — BEST MODEL")
        print(f"  Validation PR-AUC: {best_score:.6f}")
        print(f"  Best iteration   : {getattr(model, 'best_iteration', 'n/a')}")
        for param_name, param_value in best_params.items():
            print(f"  {param_name:<22}: {param_value}")
        print("-" * 70 + "\n")

    except Exception:
        logger.warning("xgboost is not available. Falling back to LogisticRegression.")
        model_type = "logistic_regression"
        model = LogisticRegression(
            class_weight="balanced",
            random_state=config.model.random_state,
            solver="saga",
            max_iter=1000,
            n_jobs=4,
        )
        model.fit(X_train_t, y_train)

    # Calibration
    calibrator = None
    if len(np.unique(y_val)) >= 2:
        try:
            method = cast(
                Literal["sigmoid", "isotonic"],
                "isotonic" if config.model.calibration_method not in {"sigmoid", "isotonic"} else config.model.calibration_method,
            )
            fitted_calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
            fitted_calibrator.fit(X_val_t, y_val)
            calibrator = fitted_calibrator
        except Exception as exc:
            calibrator = None
            logger.warning("Calibration failed, continuing with raw probabilities: %s", exc)
    else:
        logger.warning("Calibration skipped: validation split has a single class.")

    train_raw_probs = np.asarray(model.predict_proba(X_train_t)[:, 1], dtype=float)
    val_raw_probs = np.asarray(model.predict_proba(X_val_t)[:, 1], dtype=float)
    test_raw_probs = np.asarray(model.predict_proba(X_test_t)[:, 1], dtype=float)

    if calibrator is None:
        train_probs = train_raw_probs
        val_probs = val_raw_probs
        test_probs = test_raw_probs
    else:
        train_cal_probs = np.asarray(calibrator.predict_proba(X_train_t)[:, 1], dtype=float)
        val_cal_probs = np.asarray(calibrator.predict_proba(X_val_t)[:, 1], dtype=float)
        test_cal_probs = np.asarray(calibrator.predict_proba(X_test_t)[:, 1], dtype=float)

        train_probs = np.where(np.isfinite(train_cal_probs), train_cal_probs, train_raw_probs)
        val_probs = np.where(np.isfinite(val_cal_probs), val_cal_probs, val_raw_probs)
        test_probs = np.where(np.isfinite(test_cal_probs), test_cal_probs, test_raw_probs)

    if (
        (not np.isfinite(train_probs).all())
        or (not np.isfinite(val_probs).all())
        or (not np.isfinite(test_probs).all())
    ):
        logger.warning("Probabilities contain non-finite values; falling back to raw model probabilities.")
        train_probs = train_raw_probs
        val_probs = val_raw_probs
        test_probs = test_raw_probs

    if not np.isfinite(test_probs).all():
        raise RuntimeError("Model produced non-finite test probabilities after fallback; refusing to persist artefacts.")

    threshold_train_opt = _compute_best_f1_threshold(y_train, train_probs)
    threshold_val_opt = _compute_best_f1_threshold(y_val, val_probs)
    threshold_test_opt = _compute_best_f1_threshold(y_test, test_probs)

    # Single operating threshold used for evaluation/inference decisions.
    # We intentionally select this from validation data.
    threshold = threshold_val_opt

    train_metrics = _evaluate(y_train, train_probs, threshold)
    val_metrics = _evaluate(y_val, val_probs, threshold)
    test_metrics = _evaluate(y_test, test_probs, threshold)

    metrics = {
        "threshold": float(threshold),
        "threshold_source": "validation_f1_optimal",
        "train_optimal_threshold": float(threshold_train_opt),
        "val_optimal_threshold": float(threshold_val_opt),
        "test_optimal_threshold": float(threshold_test_opt),
        "pr_auc": float(test_metrics["pr_auc"]),
        "recall": float(test_metrics["recall"]),
        "roc_auc": float(test_metrics["roc_auc"]),
        "precision": float(test_metrics["precision"]),
    }
    metrics.update(_prefix_metrics(train_metrics, "train"))
    metrics.update(_prefix_metrics(val_metrics, "val"))
    metrics.update(_prefix_metrics(test_metrics, "test"))

    # Permutation importance
    X_test_for_perm: Any = X_test_t
    if hasattr(X_test_t, "toarray"):
        X_test_for_perm = np.asarray(cast(Any, X_test_t).toarray())

    perm = permutation_importance(model, X_test_for_perm, y_test, n_repeats=5, random_state=config.model.random_state)
    feature_importance_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    try:
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
        importances = np.array(perm["importances_mean"])
        importance_stds = np.array(perm["importances_std"])

        feature_importance_df = pd.DataFrame({
            "feature": transformed_feature_names,
            "importance_mean": importances,
            "importance_std": importance_stds,
        }).sort_values(by="importance_mean", ascending=False, kind="mergesort").reset_index(drop=True)

        ranking = np.argsort(importances)[::-1][:15]
        metrics["top_global_features"] = [
            {"feature": transformed_feature_names[idx], "importance": float(importances[idx])}
            for idx in ranking
        ]
    except Exception:
        metrics["top_global_features"] = []

    # Persist artefacts
    artefact_bundle = {
        "model": model,
        "model_type": model_type,
        "calibrator": calibrator,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "threshold": threshold,
        "metrics": metrics,
        "config": asdict(config.model),
    }

    model_path = config.paths.artefacts_dir / "grid_cv_model_bundle.joblib"
    metrics_path = config.paths.artefacts_dir / "grid_cv_train_metrics.json"
    metrics_csv_path = config.paths.artefacts_dir / "grid_cv_train_metrics.csv"
    feature_importance_path = config.paths.artefacts_dir / "grid_cv_feature_importance.csv"
    feature_frame_path = config.paths.artefacts_dir / "grid_cv_training_features.parquet"

    joblib.dump(artefact_bundle, model_path)
    feature_frame.to_parquet(feature_frame_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_csv = pd.DataFrame({
        key: [value]
        for key, value in metrics.items()
        if not isinstance(value, (list, dict))
    })
    metrics_csv.to_csv(metrics_csv_path, index=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        gcs_store.upload_file(model_path, remote_name=f"artefacts/{model_path.name}")
        gcs_store.upload_file(metrics_path, remote_name=f"artefacts/{metrics_path.name}")
        gcs_store.upload_file(metrics_csv_path, remote_name=f"artefacts/{metrics_csv_path.name}")
        gcs_store.upload_file(feature_importance_path, remote_name=f"artefacts/{feature_importance_path.name}")
        gcs_store.upload_file(feature_frame_path, remote_name=f"artefacts/{feature_frame_path.name}")

    logger.info("Training complete. model_path=%s metrics=%s", model_path, metrics)

    return TrainOutputs(
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        metrics_csv_path=metrics_csv_path,
        feature_importance_path=feature_importance_path,
        feature_frame_path=feature_frame_path,
    )
