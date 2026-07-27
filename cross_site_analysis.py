#!/usr/bin/env python3
"""Compute same-date cross-site Spearman correlations and range retention.

Percentage range retention is calculated only for SNAP and Copernicus because these products are expressed in nominal LAI units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METHODS = ["NDVI", "NDWI", "SAVI", "OSAVI", "EVI", "NDRE", "SNAP", "Copernicus"]
LAI_PRODUCTS = ["SNAP", "Copernicus"]
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
    parser.add_argument(
        "--min-sites",
        type=int,
        default=4,
        help="Minimum number of matched sites required for a same-date result.",
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


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.workbook)

    spearman_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    range_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for field_date, group in data.groupby("Field_Date", sort=True):
        for method in METHODS:
            pair = group[["Field_LAI", method]].dropna()
            if len(pair) < args.min_sites:
                continue

            raw_rho, raw_p = safe_spearman(pair[method], pair["Field_LAI"])
            multiplier = -1.0 if method == "NDWI" else 1.0
            oriented_rho, oriented_p = safe_spearman(
                multiplier * pair[method], pair["Field_LAI"]
            )
            spearman_rows.append(
                {
                    "Field_Date": field_date,
                    "Method": method,
                    "n": len(pair),
                    "Raw_Spearman_rho": raw_rho,
                    "Raw_p": raw_p,
                    "Orientation_multiplier": multiplier,
                    "Oriented_Spearman_rho": oriented_rho,
                    "Oriented_p": oriented_p,
                }
            )

        for product in LAI_PRODUCTS:
            pair = group[["Field_LAI", product]].dropna()
            if len(pair) < args.min_sites:
                continue
            field_range = float(pair["Field_LAI"].max() - pair["Field_LAI"].min())
            product_range = float(pair[product].max() - pair[product].min())
            retention = np.nan if field_range == 0 else 100.0 * product_range / field_range
            range_rows.append(
                {
                    "Field_Date": field_date,
                    "Product": product,
                    "n": len(pair),
                    "Field_LAI_Range": field_range,
                    "Product_LAI_Range": product_range,
                    "Range_Retention_Percent": retention,
                }
            )

    spearman = pd.DataFrame(spearman_rows)
    range_retention = pd.DataFrame(range_rows)

    spearman.to_csv(args.outdir / "same_date_cross_site_spearman.csv", index=False)
    range_retention.to_csv(args.outdir / "range_retention_percent.csv", index=False)

    print("Direction-oriented same-date cross-site Spearman correlations:")
    print(
        spearman.pivot(index="Method", columns="Field_Date", values="Oriented_Spearman_rho")
        .round(2)
        .to_string()
    )
    print("\nPercentage range retention:")
    print(
        range_retention.pivot(
            index="Field_Date", columns="Product", values="Range_Retention_Percent"
        )
        .round(1)
        .to_string()
    )
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
