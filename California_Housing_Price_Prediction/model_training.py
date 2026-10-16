from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ----------------------------- Configuration -----------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "median_house_value"

BASE_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = (
    BASE_DIR / "housing(3).csv",
    BASE_DIR / "housing.csv",
    BASE_DIR / "housing_raw.csv",
)

MODEL_PATH = BASE_DIR / "best_housing_model.joblib"
METRICS_PATH = BASE_DIR / "model_comparison.csv"
SUMMARY_PATH = BASE_DIR / "training_summary.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------- Utilities -------------------------------

def find_input_file() -> Path:
    """Find a supported raw housing CSV beside this script."""
    for candidate in INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate

    names = "\n".join(f"  - {p.name}" for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the raw housing dataset beside model_training.py.\n"
        f"Looked for:\n{names}\n"
        "Place your raw CSV in the same folder as this script."
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the raw dataset."""
    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(f"The dataset is empty: {path}")

    # Target values must be numeric and present for supervised learning.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    missing_target = int(df[TARGET].isna().sum())
    if missing_target:
        logger.warning(
            "Dropping %d row(s) with missing/non-numeric target values.",
            missing_target,
        )
        df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ValueError("No usable rows remain after validating the target.")

    # Remove exact duplicate observations before splitting.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate row(s).", removed)

    return df


def make_stratification_labels(
    X: pd.DataFrame,
) -> pd.Series | None:
    """
    Create median-income bins for a representative split.
    Returns None if there are too few values or any bin would be too small.
    """
    if "median_income" not in X.columns:
        return None

    income = pd.to_numeric(X["median_income"], errors="coerce")
    try:
        bins = pd.qcut(income, q=5, labels=False, duplicates="drop")
        counts = bins.value_counts(dropna=False)
        if bins.isna().any() or bins.nunique() < 2 or counts.min() < 2:
            return None
        # Each stratum needs enough examples to appear in both train and test.
        if counts.min() < 2:
            return None
        return bins
    except (ValueError, TypeError):
        return None


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build transformations to be fitted only on the training partition."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_models() -> Dict[str, object]:
    """Return a small, useful baseline/model-comparison suite."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """Calculate standard regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------- Main -----------------------------------

def main() -> None:
    input_path = find_input_file()
    logger.info("Loading dataset: %s", input_path.name)

    df = load_dataset(input_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after separating the target.")

    stratify_labels = make_stratification_labels(X)
    if stratify_labels is not None:
        logger.info("Using median_income groups for stratified train/test split.")
    else:
        logger.warning(
            "Could not create reliable income strata; using a random split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    logger.info(
        "Rows: %d | Features: %d | Train: %d | Test: %d",
        len(df), X.shape[1], len(X_train), len(X_test),
    )

    results = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, model in build_models().items():
        logger.info("Training %s ...", name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = regression_metrics(y_test, predictions)
            row = {"Model": name, **metrics, "Status": "OK"}
            results.append(row)
            fitted_pipelines[name] = pipeline
            logger.info(
                "%s | RMSE: %.2f | MAE: %.2f | R²: %.4f",
                name, metrics["RMSE"], metrics["MAE"], metrics["R2"],
            )
        except Exception as exc:
            # Keep the comparison running if a single estimator fails.
            logger.exception("Model '%s' failed: %s", name, exc)
            results.append(
                {
                    "Model": name,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Status": f"FAILED: {type(exc).__name__}: {exc}",
                }
            )

    successful = [
        row for row in results
        if row["Status"] == "OK" and np.isfinite(row["RMSE"])
    ]
    if not successful:
        raise RuntimeError(
            "All models failed. Review the error messages above and check "
            "the input dataset and installed scikit-learn version."
        )

    comparison = pd.DataFrame(results).sort_values(
        by="RMSE",
        ascending=True,
        na_position="last",
    )
    comparison.to_csv(METRICS_PATH, index=False)

    best_row = min(successful, key=lambda row: row["RMSE"])
    best_name = best_row["Model"]
    best_pipeline = fitted_pipelines[best_name]

    # Save the full fitted preprocessing + estimator pipeline.
    # It accepts raw feature rows, not already-scaled CSV rows.
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "target": TARGET,
        "feature_columns": X.columns.tolist(),
        "metrics": {
            key: best_row[key] for key in ("MAE", "MSE", "RMSE", "R2")
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_filename": input_path.name,
    }
    joblib.dump(artifact, MODEL_PATH)

    summary = {
        "input_file": str(input_path),
        "rows_after_deduplication": int(len(df)),
        "feature_count": int(X.shape[1]),
        "target": TARGET,
        "best_model": best_name,
        "selection_metric": "RMSE (lower is better)",
        "best_model_metrics": artifact["metrics"],
        "model_artifact": str(MODEL_PATH),
        "comparison_csv": str(METRICS_PATH),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON (sorted by RMSE; lower is better)")
    print("=" * 72)
    print(
        comparison[["Model", "MAE", "RMSE", "R2", "Status"]]
        .to_string(index=False, float_format=lambda value: f"{value:,.4f}")
    )
    print("\n" + "=" * 72)
    print(f"Best model: {best_name}")
    print(f"MAE : {best_row['MAE']:,.2f}")
    print(f"MSE : {best_row['MSE']:,.2f}")
    print(f"RMSE: {best_row['RMSE']:,.2f}")
    print(f"R²  : {best_row['R2']:.4f}")
    print("=" * 72)
    print(f"Saved fitted model pipeline: {MODEL_PATH}")
    print(f"Saved comparison table:      {METRICS_PATH}")
    print(f"Saved training summary:      {SUMMARY_PATH}")
    print("\nNote: metrics are from the held-out test set; no claim is made")
    print("that these results generalize perfectly to future housing data.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error("%s", error)
        sys.exit(1)
