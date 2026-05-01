from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "industrial_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "production_model.pkl"

DATE_COLUMNS = ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]
NUMERIC_FEATURES = [
    "Material_Used",
    "Processing_Time",
    "Energy_Consumption",
    "Machine_Availability",
    "Scheduled_Hour",
    "Scheduled_DayOfWeek",
    "Weekend_Flag",
]
CATEGORICAL_FEATURES = ["Machine_ID", "Operation_Type", "Shift_Type"]
CONTROLLABLE_FEATURES = [
    "Material_Used",
    "Processing_Time",
    "Energy_Consumption",
    "Machine_Availability",
]
TARGET_COLUMN = "Production_Output_Index"


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _shift_label(hour: float) -> str:
    if pd.isna(hour):
        return "Unknown"
    if 6 <= hour < 14:
        return "Morning"
    if 14 <= hour < 22:
        return "Afternoon"
    return "Night"


def _machine_bonus(machine_id: str) -> float:
    mapping = {"M01": 1.8, "M02": 1.2, "M03": 0.9, "M04": 1.5, "M05": 1.0}
    return mapping.get(str(machine_id), 0.8)


def _operation_bonus(operation_type: str) -> float:
    mapping = {"Additive": 4.2, "Grinding": 2.8, "Lathe": 2.2, "Milling": 3.6}
    return mapping.get(str(operation_type), 2.5)


def _shift_bonus(shift_type: str) -> float:
    mapping = {"Morning": 4.0, "Afternoon": 2.0, "Night": -2.0, "Unknown": 0.0}
    return mapping.get(str(shift_type), 0.0)


def _build_proxy_target(df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(42)
    base_score = (
        56.0
        + 0.38 * df["Machine_Availability"].astype(float)
        - 0.28 * df["Processing_Time"].astype(float)
        - 0.85 * df["Energy_Consumption"].astype(float)
        + 1.35 * df["Material_Used"].astype(float)
        + df["Machine_ID"].map(_machine_bonus).astype(float)
        + df["Operation_Type"].map(_operation_bonus).astype(float)
        + df["Shift_Type"].map(_shift_bonus).astype(float)
        + np.where(df["Weekend_Flag"] == 1, -1.5, 0.75)
    )
    noise = rng.normal(0, 1.75, len(df))
    return pd.Series(np.clip(base_score + noise, 0, 100), index=df.index, name=TARGET_COLUMN)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Scheduled_Hour"] = prepared["Scheduled_Start"].dt.hour.fillna(0).astype(int)
    prepared["Scheduled_DayOfWeek"] = prepared["Scheduled_Start"].dt.dayofweek.fillna(0).astype(int)
    prepared["Weekend_Flag"] = prepared["Scheduled_DayOfWeek"].isin([5, 6]).astype(int)
    prepared["Shift_Type"] = prepared["Scheduled_Hour"].apply(_shift_label)
    prepared["Schedule_Duration_Minutes"] = (
        (prepared["Scheduled_End"] - prepared["Scheduled_Start"]).dt.total_seconds() / 60.0
    ).fillna(prepared["Processing_Time"].astype(float))
    prepared["Start_Delay_Minutes"] = (
        (prepared["Actual_Start"] - prepared["Scheduled_Start"]).dt.total_seconds() / 60.0
    )
    prepared["End_Delay_Minutes"] = (
        (prepared["Actual_End"] - prepared["Scheduled_End"]).dt.total_seconds() / 60.0
    )
    prepared["Delay_Flag"] = (prepared["Start_Delay_Minutes"].fillna(0) > 0).astype(int)
    prepared[TARGET_COLUMN] = _build_proxy_target(prepared)
    prepared.sort_values("Scheduled_Start", inplace=True)
    prepared.reset_index(drop=True, inplace=True)
    return prepared


def load_and_prepare_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return prepare_features(load_raw_data(path))


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def load_model_package(path: Path | str = MODEL_PATH) -> dict:
    return joblib.load(path)


def get_predictions(model_package: dict, frame: pd.DataFrame) -> np.ndarray:
    return model_package["pipeline"].predict(frame)


def evaluation_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def residual_control_limits(residuals: Iterable[float]) -> dict:
    residual_array = np.asarray(list(residuals), dtype=float)
    mean_residual = float(residual_array.mean())
    std_residual = float(residual_array.std(ddof=1))
    return {
        "mean": mean_residual,
        "std": std_residual,
        "ucl": mean_residual + 3 * std_residual,
        "lcl": mean_residual - 3 * std_residual,
    }


def get_coefficient_frame(model_package: dict) -> pd.DataFrame:
    pipeline = model_package["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_.ravel()
    if len(feature_names) != len(coefficients):
        feature_names = np.array([f"feature_{index}" for index in range(len(coefficients))])
    frame = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
    frame["Absolute"] = frame["Coefficient"].abs()
    return frame.sort_values("Absolute", ascending=False).reset_index(drop=True)


def build_diagnostic_frame(model_package: dict, df: pd.DataFrame) -> pd.DataFrame:
    features = get_feature_frame(df)
    predictions = get_predictions(model_package, features)
    diagnostics = df[["Job_ID", "Machine_ID", "Operation_Type", "Job_Status", "Scheduled_Start"]].copy()
    diagnostics["Actual_Output"] = df[TARGET_COLUMN].to_numpy()
    diagnostics["Predicted_Output"] = predictions
    diagnostics["Residual"] = diagnostics["Actual_Output"] - diagnostics["Predicted_Output"]
    limits = residual_control_limits(diagnostics["Residual"].to_numpy())
    diagnostics["UCL"] = limits["ucl"]
    diagnostics["LCL"] = limits["lcl"]
    diagnostics["Out_of_Control"] = ~diagnostics["Residual"].between(limits["lcl"], limits["ucl"])
    return diagnostics


def predict_with_row(model_package: dict, row: pd.Series | dict) -> float:
    frame = pd.DataFrame([row])
    return float(get_predictions(model_package, frame)[0])


def optimize_process_settings(
    model_package: dict,
    base_row: pd.Series | dict,
    bounds: dict[str, tuple[float, float]],
) -> dict:
    template = pd.Series(base_row).copy()
    initial = np.array([template[name] for name in CONTROLLABLE_FEATURES], dtype=float)

    def objective(values: np.ndarray) -> float:
        updated = template.copy()
        for index, name in enumerate(CONTROLLABLE_FEATURES):
            low, high = bounds[name]
            updated[name] = float(np.clip(values[index], low, high))
        return -predict_with_row(model_package, updated)

    scipy_bounds = [bounds[name] for name in CONTROLLABLE_FEATURES]
    result = minimize(objective, x0=initial, bounds=scipy_bounds, method="SLSQP")

    optimal = template.copy()
    for index, name in enumerate(CONTROLLABLE_FEATURES):
        low, high = bounds[name]
        optimal[name] = float(np.clip(result.x[index], low, high))

    return {
        "success": bool(result.success),
        "message": result.message,
        "optimal_row": optimal.to_dict(),
        "predicted_output": float(-result.fun),
    }


def feature_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for feature in CONTROLLABLE_FEATURES:
        low = float(df[feature].quantile(0.05))
        high = float(df[feature].quantile(0.95))
        bounds[feature] = (low, high)
    return bounds


def trend_summary(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    frame = df[["Scheduled_Start", TARGET_COLUMN]].copy()
    frame["Predicted_Output"] = predictions
    frame.rename(columns={TARGET_COLUMN: "Actual_Output"}, inplace=True)
    frame["Rolling_Actual"] = frame["Actual_Output"].rolling(window=7, min_periods=1).mean()
    frame["Rolling_Predicted"] = frame["Predicted_Output"].rolling(window=7, min_periods=1).mean()
    return frame
