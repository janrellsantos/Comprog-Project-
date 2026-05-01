from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

from dashboard.components import (
    DATA_PATH,
    MODEL_PATH,
    build_diagnostic_frame,
    feature_bounds,
    get_coefficient_frame,
    get_feature_frame,
    load_and_prepare_data,
    load_model_package,
    optimize_process_settings,
    trend_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = PROJECT_ROOT / "dashboard" / "assets" / "style.css"


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def metric_card(title: str, value: str, delta: str | None = None) -> None:
    st.metric(title, value, delta)


def build_actual_vs_predicted_figure(diagnostics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=diagnostics["Actual_Output"],
            y=diagnostics["Predicted_Output"],
            mode="markers",
            name="Jobs",
            marker=dict(color="#f4b400", size=8, opacity=0.8),
            text=diagnostics["Job_ID"],
        )
    )
    min_value = min(diagnostics["Actual_Output"].min(), diagnostics["Predicted_Output"].min())
    max_value = max(diagnostics["Actual_Output"].max(), diagnostics["Predicted_Output"].max())
    figure.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Ideal",
            line=dict(color="#5ec8c8", dash="dash"),
        )
    )
    figure.update_layout(
        title="Actual vs Predicted Production Output",
        xaxis_title="Actual Output Index",
        yaxis_title="Predicted Output Index",
        template="plotly_dark",
        height=460,
    )
    return figure


def build_trend_figure(trend: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": False}]])
    figure.add_trace(
        go.Scatter(
            x=trend["Scheduled_Start"],
            y=trend["Actual_Output"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#f4b400", width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["Scheduled_Start"],
            y=trend["Predicted_Output"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#5ec8c8", width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["Scheduled_Start"],
            y=trend["Rolling_Actual"],
            mode="lines",
            name="7-Point Rolling Actual",
            line=dict(color="#ff8a5b", width=2, dash="dot"),
        )
    )
    figure.update_layout(
        title="Forecast vs Actual Production Trend",
        xaxis_title="Scheduled Start",
        yaxis_title="Output Index",
        template="plotly_dark",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def build_control_chart(diagnostics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=diagnostics["Scheduled_Start"],
            y=diagnostics["Residual"],
            mode="lines+markers",
            name="Residual",
            line=dict(color="#9be564", width=2),
            marker=dict(color=["#ef4444" if flag else "#9be564" for flag in diagnostics["Out_of_Control"]], size=8),
        )
    )
    ucl = float(diagnostics["UCL"].iloc[0])
    lcl = float(diagnostics["LCL"].iloc[0])
    figure.add_hline(y=ucl, line_color="#ff6b6b", line_dash="dash", annotation_text="UCL")
    figure.add_hline(y=lcl, line_color="#ff6b6b", line_dash="dash", annotation_text="LCL")
    figure.add_hline(y=0.0, line_color="#94a3b8", line_dash="dot", annotation_text="Center Line")
    figure.update_layout(
        title="Residual Control Chart (3-Sigma)",
        xaxis_title="Scheduled Start",
        yaxis_title="Residual",
        template="plotly_dark",
        height=440,
    )
    return figure


def build_coefficient_figure(coefficient_frame: pd.DataFrame) -> go.Figure:
    trimmed = coefficient_frame.head(15).sort_values("Coefficient")
    figure = go.Figure(
        go.Bar(
            x=trimmed["Coefficient"],
            y=trimmed["Feature"],
            orientation="h",
            marker_color=["#5ec8c8" if value >= 0 else "#ff8a5b" for value in trimmed["Coefficient"]],
        )
    )
    figure.update_layout(
        title="Top Model Coefficients",
        xaxis_title="Coefficient Value",
        yaxis_title="Feature",
        template="plotly_dark",
        height=520,
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="Industrial Engineering Admin Dashboard", layout="wide")
    load_css()

    st.title("Industrial Engineering Production Admin Dashboard")
    st.caption("Forecasting, SPC, KPI tracking, and process optimization built from the Kaggle-style manufacturing dataset.")

    if st.sidebar.checkbox("Auto refresh every 30 seconds", value=False) and st_autorefresh is not None:
        st_autorefresh(interval=30_000, key="dashboard_refresh")

    if not MODEL_PATH.exists():
        st.warning("Model package not found. Train the model first by running python train_model.py from the project root.")
        st.stop()

    df = load_and_prepare_data(DATA_PATH)
    model_package = load_model_package(MODEL_PATH)
    full_features = get_feature_frame(df)
    full_predictions = model_package["pipeline"].predict(full_features)

    min_date = df["Scheduled_Start"].min().date()
    max_date = df["Scheduled_Start"].max().date()

    with st.sidebar:
        st.header("Control Panel")
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        machine_selection = st.multiselect("Machine IDs", sorted(df["Machine_ID"].dropna().unique()), default=sorted(df["Machine_ID"].dropna().unique()))
        operation_selection = st.multiselect("Operation Types", sorted(df["Operation_Type"].dropna().unique()), default=sorted(df["Operation_Type"].dropna().unique()))
        status_selection = st.multiselect("Job Status", sorted(df["Job_Status"].dropna().unique()), default=sorted(df["Job_Status"].dropna().unique()))
        sigma_limit = st.slider("Residual alert threshold (sigma)", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
        selected_shift = st.selectbox("Shift focus", ["All"] + sorted(df["Shift_Type"].unique().tolist()))

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered = df[
        df["Scheduled_Start"].dt.date.between(start_date, end_date)
        & df["Machine_ID"].isin(machine_selection)
        & df["Operation_Type"].isin(operation_selection)
        & df["Job_Status"].isin(status_selection)
    ].copy()
    if selected_shift != "All":
        filtered = filtered[filtered["Shift_Type"] == selected_shift]

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        st.stop()

    feature_frame = get_feature_frame(filtered)
    predictions = model_package["pipeline"].predict(feature_frame)
    diagnostics = build_diagnostic_frame(model_package, filtered)
    diagnostics["Predicted_Output"] = predictions
    diagnostics["Residual"] = diagnostics["Actual_Output"] - diagnostics["Predicted_Output"]

    residual_limits = {
        "ucl": float(diagnostics["Residual"].mean() + sigma_limit * diagnostics["Residual"].std(ddof=1)),
        "lcl": float(diagnostics["Residual"].mean() - sigma_limit * diagnostics["Residual"].std(ddof=1)),
    }
    diagnostics["UCL"] = residual_limits["ucl"]
    diagnostics["LCL"] = residual_limits["lcl"]
    diagnostics["Out_of_Control"] = ~diagnostics["Residual"].between(residual_limits["lcl"], residual_limits["ucl"])

    trend = trend_summary(filtered, predictions)
    coefficient_frame = get_coefficient_frame(model_package)

    total_jobs = len(filtered)
    avg_output = filtered["Production_Output_Index"].mean()
    stability_index = max(0.0, 1.0 - filtered["Production_Output_Index"].std(ddof=0) / max(avg_output, 1e-6))
    completion_rate = (filtered["Job_Status"] == "Completed").mean() * 100
    proxy_oee = max(
        0.0,
        (
            filtered["Machine_Availability"].mean() / 100.0
            * completion_rate / 100.0
            * (1.0 - (diagnostics["Residual"].abs().mean() / max(avg_output, 1e-6)))
        )
        * 100,
    )
    anomaly_rate = diagnostics["Out_of_Control"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Average Output", f"{avg_output:.1f}", f"{total_jobs} jobs")
    with col2:
        metric_card("Stability Index", f"{stability_index:.2f}", f"{completion_rate:.1f}% completion")
    with col3:
        metric_card("Proxy OEE", f"{proxy_oee:.1f}%", f"{filtered['Machine_Availability'].mean():.1f}% availability")
    with col4:
        metric_card("Anomaly Rate", f"{anomaly_rate:.1f}%", f"3-sigma alerts: {diagnostics['Out_of_Control'].sum()}")

    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Diagnostics", "Optimization", "Export"])

    with tab1:
        st.plotly_chart(build_trend_figure(trend), use_container_width=True)
        st.plotly_chart(build_actual_vs_predicted_figure(diagnostics), use_container_width=True)

    with tab2:
        st.plotly_chart(build_control_chart(diagnostics), use_container_width=True)
        st.plotly_chart(build_coefficient_figure(coefficient_frame), use_container_width=True)
        if diagnostics["Out_of_Control"].any():
            st.error("Process violation detected: residuals exceed the selected control limits.")
        else:
            st.success("No residuals exceeded the control limits in the selected slice.")

        alert_frame = diagnostics.loc[diagnostics["Out_of_Control"], ["Job_ID", "Machine_ID", "Operation_Type", "Job_Status", "Residual"]]
        st.subheader("Flagged Jobs")
        st.dataframe(alert_frame, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("What-if simulator")
        bounds = feature_bounds(df)
        base_row = filtered.iloc[len(filtered) // 2][feature_frame.columns].to_dict()

        with st.form("what_if_form"):
            c1, c2 = st.columns(2)
            with c1:
                material_used = st.slider("Material Used", float(bounds["Material_Used"][0]), float(bounds["Material_Used"][1]), float(base_row["Material_Used"]), step=0.01)
                processing_time = st.slider("Processing Time", float(bounds["Processing_Time"][0]), float(bounds["Processing_Time"][1]), float(base_row["Processing_Time"]), step=1.0)
            with c2:
                energy_consumption = st.slider("Energy Consumption", float(bounds["Energy_Consumption"][0]), float(bounds["Energy_Consumption"][1]), float(base_row["Energy_Consumption"]), step=0.01)
                machine_availability = st.slider("Machine Availability", float(bounds["Machine_Availability"][0]), float(bounds["Machine_Availability"][1]), float(base_row["Machine_Availability"]), step=1.0)
            submitted = st.form_submit_button("Evaluate settings")

        scenario = base_row.copy()
        scenario.update(
            {
                "Material_Used": material_used,
                "Processing_Time": processing_time,
                "Energy_Consumption": energy_consumption,
                "Machine_Availability": machine_availability,
            }
        )
        scenario_prediction = float(model_package["pipeline"].predict(pd.DataFrame([scenario]))[0])
        st.info(f"Predicted output for the selected settings: {scenario_prediction:.1f}")

        if submitted:
            optimization = optimize_process_settings(model_package, scenario, bounds)
            st.success(f"Recommended output: {optimization['predicted_output']:.1f}")
            st.json(optimization["optimal_row"])

    with tab4:
        csv_data = diagnostics.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered diagnostics CSV",
            data=csv_data,
            file_name="industrial_dashboard_filtered_data.csv",
            mime="text/csv",
        )
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)

    st.caption(
        f"Model package loaded from {MODEL_PATH.name}. Current holdout metrics: R²={model_package['metrics']['r2']:.3f}, RMSE={model_package['metrics']['rmse']:.3f}, MAE={model_package['metrics']['mae']:.3f}."
    )


if __name__ == "__main__":
    main()
