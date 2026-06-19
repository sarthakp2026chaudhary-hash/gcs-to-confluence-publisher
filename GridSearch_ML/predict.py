# predict.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from config import AppConfig, date_to_str
from extract_db import JonathonMetadataExtractor
from extract_logs_table import ErrorLogTableExtractor
from features import build_inference_features
from rca import enrich_with_rca
from storage_utils import GCSOutputStore

logger = logging.getLogger(__name__)


@dataclass
class PredictOutputs:
    predictions: pd.DataFrame
    output_csv: Path


def _validate_model_bundle(bundle: dict[str, Any]) -> None:
    required_keys = {"model", "preprocessor", "feature_cols"}
    missing = sorted(required_keys - set(bundle.keys()))
    if missing:
        raise ValueError(f"Invalid model bundle: missing required key(s): {missing}")

    feature_cols = bundle.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Invalid model bundle: 'feature_cols' must be a non-empty list.")

    model = bundle.get("model")
    preprocessor = bundle.get("preprocessor")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Invalid model bundle: 'model' does not expose predict_proba().")
    if not hasattr(preprocessor, "transform"):
        raise ValueError("Invalid model bundle: 'preprocessor' does not expose transform().")


def _load_model_bundle(config: AppConfig) -> dict[str, Any]:
    """Load model bundle using GCS-first strategy when output_gcs_uri is configured."""
    config.ensure_dirs()
    model_path = config.paths.artefacts_dir / "grid_cv_model_bundle.joblib"

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        remote_name = f"artefacts/{model_path.name}"
        try:
            logger.info("Downloading model bundle from GCS: %s", remote_name)
            gcs_store.download_file(remote_name, model_path)
        except FileNotFoundError:
            logger.warning(
                "Model bundle not found in GCS (%s); falling back to local model at %s",
                remote_name,
                model_path,
            )

    if not model_path.exists():
        raise FileNotFoundError(f"Model artefact not found: {model_path}")

    bundle = cast(dict[str, Any], joblib.load(model_path))
    _validate_model_bundle(bundle)
    return bundle


def _risk_bucket(prob: float, high: float, med: float) -> str:
    if not np.isfinite(prob):
        return "Unknown"
    if prob >= high:
        return "High"
    if prob >= med:
        return "Med"
    return "Low"


def _predict_probabilities(bundle: dict[str, Any], transformed_X: Any) -> np.ndarray:
    model = bundle["model"]
    calibrator = bundle.get("calibrator")

    base_probs = np.asarray(model.predict_proba(transformed_X)[:, 1], dtype=float)

    if calibrator is None:
        probs = base_probs
    else:
        try:
            cal_probs = np.asarray(calibrator.predict_proba(transformed_X)[:, 1], dtype=float)
            probs = np.where(np.isfinite(cal_probs), cal_probs, base_probs)
        except Exception as exc:
            logger.warning("Calibrator inference failed (%s). Falling back to raw model probabilities.", exc)
            probs = base_probs

    invalid_mask = ~np.isfinite(probs)
    if np.any(invalid_mask):
        logger.warning(
            "Model produced %s non-finite probability value(s); applying deterministic fallback.",
            int(invalid_mask.sum()),
        )
        try:
            hard_preds = np.asarray(model.predict(transformed_X), dtype=float).reshape(-1)
            hard_preds = np.clip(hard_preds, 0.0, 1.0)
            probs = np.where(invalid_mask, hard_preds, probs)
        except Exception as exc:
            logger.warning("Hard-prediction fallback failed: %s", exc)

        remaining_invalid = ~np.isfinite(probs)
        if np.any(remaining_invalid):
            finite_probs = np.asarray(probs[~remaining_invalid], dtype=float)
            fallback_prob = float(np.median(finite_probs)) if finite_probs.size else 0.5
            logger.warning(
                "Replacing %s residual non-finite probability value(s) with fallback=%.4f.",
                int(remaining_invalid.sum()),
                fallback_prob,
            )
            probs = np.where(remaining_invalid, fallback_prob, probs)

    return np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)


def _action_from_drivers(drivers: list[str]) -> str:
    text = " ".join(drivers).lower()
    if "import_error" in text or "broken_dag" in text:
        return "Validate DAG import/package dependencies and scheduler parse logs."
    if "quota" in text or "rate_limit" in text:
        return "Throttle concurrency, review quotas, and add backoff/retry controls."
    if "timeout" in text or "queue" in text:
        return "Increase timeout/worker capacity and review queue or pool saturation."
    if "oom" in text or "memory" in text:
        return "Increase memory resources or optimize task memory usage."
    if "network" in text or "permission" in text or "auth" in text:
        return "Check IAM/credentials and network connectivity to upstream systems."
    if "consecutive_failures" in text or "failure_rate" in text:
        return "Escalate DAG health check, review recent failures, and verify upstream dependencies."
    return "Review latest task logs and rerun with safeguards if dependencies are healthy."


def _resolve_risk_thresholds(
    probabilities: np.ndarray,
    configured_high: float,
    configured_med: float,
) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs)]

    if probs.size == 0:
        return configured_high, configured_med

    high = float(configured_high)
    med = float(configured_med)

    if med > high:
        med, high = high, med

    if not np.all(probs < med):
        return high, med

    p70 = float(np.quantile(probs, 0.70))
    p90 = float(np.quantile(probs, 0.90))

    med_fallback = max(0.0, min(1.0, p70))
    high_fallback = max(med_fallback + 1e-6, min(1.0, p90))

    logger.warning(
        "Configured thresholds high=%.3f med=%.3f classify all predictions as Low. "
        "Applying fallback thresholds high=%.3f med=%.3f (p90/p70).",
        high,
        med,
        high_fallback,
        med_fallback,
    )
    return high_fallback, med_fallback


def _top_drivers_from_contribs(
    bundle: dict[str, Any],
    transformed_X: Any,
    top_k: int = 3,
) -> list[list[str]]:
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    try:
        dmat = DMatrix(transformed_X, feature_names=feature_names.tolist())
        contribs = np.array(model.get_booster().predict(dmat, pred_contribs=True))

        drivers: list[list[str]] = []
        for row in contribs:
            row_wo_bias = row[:-1]
            idx = np.argsort(np.abs(row_wo_bias))[::-1][:top_k]
            drivers.append([str(feature_names[i]) for i in idx])
        return drivers

    except Exception as exc:
        logger.warning("Top-driver extraction failed (%s); using generic placeholders.", exc)
        return [["feature_signal_unavailable"] * min(top_k, 1) for _ in range(transformed_X.shape[0])]


def predict_for_date(config: AppConfig, target_date: date) -> PredictOutputs:
    bundle = _load_model_bundle(config)

    extractor = JonathonMetadataExtractor(config)
    extracted = extractor.fetch(start_date=None, end_date=target_date)

    run_log_features = pd.DataFrame()
    if config.logs.enabled and config.logs.error_log_table:
        try:
            run_log_features = ErrorLogTableExtractor(config.logs).extract_for_runs(
                extracted.dag_run,
                extracted.task_instance,
            )
        except Exception as exc:
            logger.warning("Log extraction failed for inference, continuing without logs: %s", exc)

    inference = build_inference_features(
        target_date=target_date,
        dag_run_df=extracted.dag_run,
        task_instance_df=extracted.task_instance,
        dag_df=extracted.dag,
        run_log_features_df=run_log_features,
        cfg=config.features,
    )

    frame = inference.feature_frame

    if frame.empty:
        raise RuntimeError(f"No inference features produced for date {target_date}.")

    feature_cols = cast(list[str], bundle["feature_cols"])

    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan

    X = frame[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.where(pd.notna(X), np.nan)

    transformed_X = bundle["preprocessor"].transform(X)

    probs = _predict_probabilities(bundle, transformed_X)

    finite_probs = probs[np.isfinite(probs)]
    if finite_probs.size:
        logger.info(
            "Probability summary: rows=%s min=%.4f p50=%.4f p90=%.4f max=%.4f",
            len(probs),
            float(np.min(finite_probs)),
            float(np.quantile(finite_probs, 0.5)),
            float(np.quantile(finite_probs, 0.9)),
            float(np.max(finite_probs)),
        )
    else:
        logger.warning("Probability summary unavailable: all predicted probabilities are non-finite.")

    resolved_high, resolved_med = _resolve_risk_thresholds(
        probs,
        config.model.high_risk_threshold,
        config.model.medium_risk_threshold,
    )

    decision_threshold = float(bundle.get("threshold", config.model.decision_threshold))

    top_driver_lists = _top_drivers_from_contribs(bundle, transformed_X, top_k=3)

    pred_df = frame[["dag_id", "run_id", "logical_date"]].copy().reset_index(drop=True)
    pred_df["predicted_failure_probability"] = probs
    pred_df["risk_bucket"] = pred_df["predicted_failure_probability"].apply(
        lambda p: _risk_bucket(float(p), resolved_high, resolved_med)
    )
    pred_df["predicted_fail_flag"] = pd.Series((probs > decision_threshold).astype(int), dtype="Int64")
    pred_df["top_drivers"] = ["|".join(drivers) for drivers in top_driver_lists]
    pred_df["suggested_action"] = [_action_from_drivers(drivers) for drivers in top_driver_lists]

    pred_df = pred_df.sort_values(by="predicted_failure_probability", ascending=False).reset_index(drop=True)
    pred_df = cast(pd.DataFrame, pred_df)

    pred_df = enrich_with_rca(pred_df, config.logs, dag_run_df=extracted.dag_run)

    out_name = f"grid_cv_predictions_{date_to_str(target_date)}.csv"
    output_csv = config.paths.outputs_dir / out_name

    pred_df.to_csv(output_csv, index=False)

    if config.paths.output_gcs_uri:
        gcs_store = GCSOutputStore(config.paths.output_gcs_uri)
        gcs_store.upload_file(output_csv, remote_name=f"outputs/{output_csv.name}")

    logger.info(
        "Predictions saved to %s (%s rows). threshold=%.4f risk_thresholds=[med=%.4f, high=%.4f]",
        output_csv,
        len(pred_df),
        decision_threshold,
        resolved_med,
        resolved_high,
    )

    return PredictOutputs(predictions=cast(pd.DataFrame, pred_df), output_csv=output_csv)
