from __future__ import annotations

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dashboard.components import (
    CATEGORICAL_FEATURES,
    CONTROLLABLE_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    get_feature_frame,
    load_and_prepare_data,
)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def train_and_save() -> dict:
    df = load_and_prepare_data()
    feature_frame = get_feature_frame(df)
    target = df[TARGET_COLUMN].copy()

    split_index = int(len(df) * 0.8)
    X_train = feature_frame.iloc[:split_index]
    X_test = feature_frame.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }

    residuals = y_test.to_numpy() - predictions
    residual_mean = float(residuals.mean())
    residual_std = float(residuals.std(ddof=1))

    package = {
        "pipeline": pipeline,
        "metrics": metrics,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "control_limits": {
            "ucl": residual_mean + 3 * residual_std,
            "lcl": residual_mean - 3 * residual_std,
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "controllable_features": CONTROLLABLE_FEATURES,
        "target_column": TARGET_COLUMN,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODEL_PATH)
    return package


if __name__ == "__main__":
    result = train_and_save()
    print("Saved model package to", MODEL_PATH)
    print(result["metrics"])
