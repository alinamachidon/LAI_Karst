#!/usr/bin/env python3
"""Compute pooled linear relationships and within-site Pearson correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

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


def safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.workbook)

    pooled_rows: list[dict[str, float | int | str]] = []
    within_rows: list[dict[str, float | int | str]] = []

    for method in METHODS:
        matched = data[["Field_LAI", method]].dropna()
        result = linregress(matched[method], matched["Field_LAI"])
        pooled_rows.append(
            {
                "Method": method,
                "n": len(matched),
                "Slope_Field_on_Method": result.slope,
                "Intercept": result.intercept,
                "Pearson_r": result.rvalue,
                "Pearson_p": result.pvalue,
                "R2": result.rvalue**2,
                "Slope_SE": result.stderr,
            }
        )

        for site, group in data.groupby("Site", sort=False):
            pair = group[["Field_LAI", method]].dropna()
            r_value, p_value = safe_pearson(pair[method], pair["Field_LAI"])
            within_rows.append(
                {
                    "Method": method,
                    "Site": site,
                    "n": len(pair),
                    "Pearson_r": r_value,
                    "Pearson_p": p_value,
                }
            )

    pooled = pd.DataFrame(pooled_rows)
    within = pd.DataFrame(within_rows)

    pooled.to_csv(args.outdir / "pooled_linear_regression.csv", index=False)
    within.to_csv(args.outdir / "within_site_pearson_correlations.csv", index=False)

    print("Pooled linear relationships (Field LAI as response):")
    print(
        pooled[["Method", "n", "Slope_Field_on_Method", "Intercept", "Pearson_r", "R2"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nWithin-site Pearson correlations:")
    print(
        within.pivot(index="Method", columns="Site", values="Pearson_r")
        .round(2)
        .to_string()
    )
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
