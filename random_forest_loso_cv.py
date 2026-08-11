#!/usr/bin/env python3
"""Random Forest full-sample and leave-one-site-out cross-validation analysis.

This script extends the manuscript's univariate Random Forest analysis by
adding site-blocked leave-one-site-out cross-validation (LOSO-CV). In each
fold, all observations from one field site are withheld and the model is
trained on observations from the remaining sites.

Expected input: LAI_data.xlsx with sheet 'All_Data_Long' and columns:
Field_Date, Site, Field_LAI, NDVI, NDWI, SAVI, OSAVI, EVI, NDRE, SNAP,
Copernicus.
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


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0,
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.workbook)

    metric_rows: list[dict[str, float | int | str]] = []
    fold_metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for method in METHODS:
        matched = data[["Field_Date", "Site", "Field_LAI", method]].dropna().copy()
        matched = matched.sort_values(["Site", "Field_Date"]).reset_index(drop=True)

        # Apparent full-sample fit, retained for comparison with the previous analysis.
        x = matched[[method]].to_numpy(dtype=float)
        y = matched["Field_LAI"].to_numpy(dtype=float)
        full_model = make_model()
        full_model.fit(x, y)
        full_pred = full_model.predict(x)

        # Leave-one-site-out cross-validation.
        loso_pred = np.full(len(matched), np.nan, dtype=float)
        for held_out_site in sorted(matched["Site"].unique()):
            test_mask = matched["Site"] == held_out_site
            train = matched.loc[~test_mask]
            test = matched.loc[test_mask]

            model = make_model()
            model.fit(train[[method]].to_numpy(dtype=float), train["Field_LAI"].to_numpy(dtype=float))
            pred = model.predict(test[[method]].to_numpy(dtype=float))
            loso_pred[test.index.to_numpy()] = pred

            fold_metric_rows.append(
                {
                    "Method": method,
                    "Held_out_site": held_out_site,
                    "n_test": len(test),
                    "MAE_LOSO_fold": mean_absolute_error(test["Field_LAI"], pred),
                    "MSE_LOSO_fold": mean_squared_error(test["Field_LAI"], pred),
                    "R2_LOSO_fold": safe_r2(test["Field_LAI"].to_numpy(dtype=float), pred),
                }
            )

        metric_rows.append(
            {
                "Method": method,
                "n": len(matched),
                "MAE_full": mean_absolute_error(y, full_pred),
                "MSE_full": mean_squared_error(y, full_pred),
                "R2_full": r2_score(y, full_pred),
                "MAE_LOSO": mean_absolute_error(y, loso_pred),
                "MSE_LOSO": mean_squared_error(y, loso_pred),
                "R2_LOSO": r2_score(y, loso_pred),
            }
        )

        for row, pred_full, pred_loso in zip(matched.itertuples(index=False), full_pred, loso_pred):
            prediction_rows.append(
                {
                    "Field_Date": row.Field_Date,
                    "Site": row.Site,
                    "Method": method,
                    "Field_LAI": row.Field_LAI,
                    "Predictor": getattr(row, method),
                    "Predicted_Field_LAI_full": pred_full,
                    "Residual_full": pred_full - row.Field_LAI,
                    "Predicted_Field_LAI_LOSO": pred_loso,
                    "Residual_LOSO": pred_loso - row.Field_LAI,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.DataFrame(prediction_rows)

    metrics.to_csv(args.outdir / "random_forest_loso_metrics.csv", index=False)
    fold_metrics.to_csv(args.outdir / "random_forest_loso_fold_metrics.csv", index=False)
    predictions.to_csv(args.outdir / "random_forest_loso_predictions.csv", index=False)

    print("Random Forest apparent full-sample fit and LOSO-CV:")
    print(
        metrics.round(
            {
                "MAE_full": 2,
                "MSE_full": 2,
                "R2_full": 2,
                "MAE_LOSO": 2,
                "MSE_LOSO": 2,
                "R2_LOSO": 2,
            }
        ).to_string(index=False)
    )
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
