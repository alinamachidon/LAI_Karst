#!/usr/bin/env python3
"""Reproduce the manuscript's full-sample Random Forest fit.

The same observations are used to fit and evaluate each univariate model.
The resulting MAE, MSE, and R² values describe apparent in-sample fit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METHODS = ["NDVI", "NDWI", "SAVI", "OSAVI", "EVI", "NDRE", "SNAP", "Copernicus"]
REQUIRED_COLUMNS = ["Field_Date", "Site", "Field_LAI", *METHODS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("LAI_data.xlsx"),
        help="Input workbook containing the All_Data_Long sheet.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV outputs.",
    )
    return parser.parse_args()


def load_data(workbook: Path) -> pd.DataFrame:
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    data = pd.read_excel(
        workbook,
        sheet_name="All_Data_Long",
        engine="openpyxl",
        parse_dates=["Field_Date", "Acquisition_Date"],
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in All_Data_Long: {missing}")
    return data


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.workbook)

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for method in METHODS:
        matched = data[["Field_Date", "Site", "Field_LAI", method]].dropna().copy()
        x = matched[[method]].to_numpy(dtype=float)
        y = matched["Field_LAI"].to_numpy(dtype=float)

        model = RandomForestRegressor(
            n_estimators=100,
            criterion="squared_error",
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=0,
        )
        model.fit(x, y)
        predicted = model.predict(x)

        metric_rows.append(
            {
                "Method": method,
                "n": len(y),
                "MAE": mean_absolute_error(y, predicted),
                "MSE": mean_squared_error(y, predicted),
                "R2": r2_score(y, predicted),
            }
        )

        for (_, row), prediction in zip(matched.iterrows(), predicted):
            prediction_rows.append(
                {
                    "Field_Date": row["Field_Date"],
                    "Site": row["Site"],
                    "Method": method,
                    "Field_LAI": row["Field_LAI"],
                    "Predictor": row[method],
                    "Predicted_Field_LAI": prediction,
                    "Residual": prediction - row["Field_LAI"],
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)

    metrics.to_csv(args.outdir / "random_forest_full_sample_metrics.csv", index=False)
    predictions.to_csv(args.outdir / "random_forest_full_sample_predictions.csv", index=False)

    print(" Random Forest fit:")
    print(metrics.round({"MAE": 2, "MSE": 2, "R2": 2}).to_string(index=False))
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
