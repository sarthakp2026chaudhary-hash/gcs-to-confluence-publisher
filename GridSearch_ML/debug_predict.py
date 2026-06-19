# debug_predict.py

"""Quick diagnostic to test probability computation in predict pipeline."""

import sys
from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from config import load_config, get_prediction_date

try:
    config = load_config()
    logger.info("Config loaded successfully")

    # Test with yesterday's date if needed
    target_date = get_prediction_date(None)
    logger.info(f"Testing prediction for: {target_date}")

    from predict import predict_for_date

    result = predict_for_date(config, target_date)
    pred_df = result.predictions

    logger.info(f"Prediction completed. Rows: {len(pred_df)}")
    logger.info(f"\nDataFrame columns: {pred_df.columns.tolist()}")
    logger.info("\nFirst 5 rows:")
    print(
        pred_df[["dag_id", "run_id", "predicted_failure_probability", "risk_bucket", "predicted_failure_flag"]].head()
    )

    # Check for NaN counts
    for col in ["predicted_failure_probability", "risk_bucket", "predicted_failure_flag"]:
        nan_count = pred_df[col].isna().sum()
        logger.info(f"\n{col} NaN count: {nan_count}/{len(pred_df)}")
        if col == "predicted_failure_probability":
            non_null = pred_df[col].dropna()
            if len(non_null) > 0:
                logger.info(f"  Min: {non_null.min():.4f}, Max: {non_null.max():.4f}, Mean: {non_null.mean():.4f}")

    logger.info(f"\nRisk bucket distribution:\n{pred_df['risk_bucket'].value_counts()}")
    logger.info(f"\nPredicted failure flag distribution:\n{pred_df['predicted_failure_flag'].value_counts()}")

except Exception as exc:
    logger.error(f"Error during prediction: {exc}", exc_info=True)
    sys.exit(1)
