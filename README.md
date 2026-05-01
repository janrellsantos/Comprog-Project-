# Industrial Engineering Production Admin Dashboard

This project builds an interactive industrial dashboard from the Kaggle-style dataset in `data/industrial_data.csv`.

The source CSV does not include a native continuous production target, so the project derives a documented proxy KPI named `Production_Output_Index` from the available process signals. That keeps the regression model, control charts, and optimization module consistent with the dataset that is actually available.

## What is included

- `notebooks/01_eda_analysis.ipynb` for exploratory analysis, outliers, and correlations
- `notebooks/02_model_training.ipynb` for feature engineering, model training, and diagnostics
- `dashboard/app.py` for the Streamlit dashboard
- `dashboard/components.py` for shared data, model, chart, and optimization helpers
- `models/production_model.pkl` for the serialized model package

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model package:

```powershell
python train_model.py
```

4. Launch the dashboard:

```powershell
cd dashboard
streamlit run app.py
```

## Dashboard features

- KPI cards for average output, stability, completion rate, and proxy OEE
- Forecast vs actual trend chart with confidence bands
- Actual vs predicted scatter plot
- Residual control chart with 3-sigma violations
- Feature coefficient bar chart
- Process anomaly table and exportable filtered data
- What-if optimization panel for controllable numeric process inputs

## Notes

- The dashboard uses scheduled timestamps to create time-aware features.
- Control chart alerts are based on model residuals and 3-sigma limits.
- If you swap in a different Kaggle industrial dataset, update the feature list and target builder in `dashboard/components.py`.
