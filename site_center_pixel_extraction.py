import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
import re

from pathlib import Path
from datetime import datetime


# ============================================================
# USER SETTINGS
# ============================================================

# Point shapefile containing the sampling-site coordinates
SHAPEFILE_PATH = Path("LAI2021_PPCS.shp")

# Metadata containing:
# Station, Latitude, Longitude
METADATA_PATH = Path("point_metadata.csv")

# Folder containing the LAI GeoTIFF files.
# "." means the same folder in which the script is run.
RASTER_FOLDER = Path(".")

# Output folder
OUTPUT_FOLDER = Path("lai_outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Scaling factor of the raster.

LAI_SCALE_FACTOR = 0.0001

# Number of decimal places in output LAI
LAI_DECIMALS = 2


# ============================================================
# 1. LOAD FIELD-SITE POINTS
# ============================================================

print("Loading field-site shapefile...")

points_gdf = gpd.read_file(SHAPEFILE_PATH)

if points_gdf.empty:
    raise ValueError("The shapefile contains no points.")

if points_gdf.crs is None:
    raise ValueError(
        "The shapefile has no CRS defined. "
        "A valid CRS is required before raster extraction."
    )

print(f"Number of points in shapefile: {len(points_gdf)}")
print(f"Original point CRS: {points_gdf.crs}")


# ============================================================
# 2. LOAD METADATA
# ============================================================

print("Loading point metadata...")

meta_df = pd.read_csv(METADATA_PATH)

required_columns = {"Station", "Latitude", "Longitude"}

missing_columns = required_columns - set(meta_df.columns)

if missing_columns:
    raise ValueError(
        f"Metadata file is missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# 3. MATCH SHAPEFILE POINTS WITH STATION IDs
# ============================================================

# Transform temporarily to WGS84 because metadata contains
# latitude/longitude coordinates.
points_wgs = points_gdf.to_crs(epsg=4326).copy()

# Give every point a stable internal ID BEFORE doing the merge.
# This avoids relying on DataFrame row order later.
points_wgs["point_index"] = np.arange(len(points_wgs))

# Round coordinates to match metadata.
points_wgs["Latitude"] = points_wgs.geometry.y.round(5)
points_wgs["Longitude"] = points_wgs.geometry.x.round(5)

meta_df = meta_df.copy()
meta_df["Latitude"] = meta_df["Latitude"].round(5)
meta_df["Longitude"] = meta_df["Longitude"].round(5)

# We only need the station information from metadata here.
station_lookup = meta_df[
    ["Station", "Latitude", "Longitude"]
].copy()

matched = points_wgs.merge(
    station_lookup,
    how="left",
    on=["Latitude", "Longitude"]
)


# Check whether every shapefile point found a station
unmatched = matched[matched["Station"].isna()]

if not unmatched.empty:
    print("\nWARNING: Some points could not be matched to metadata:")
    print(
        unmatched[
            ["point_index", "Latitude", "Longitude"]
        ].to_string(index=False)
    )

    raise ValueError(
        "At least one shapefile point could not be matched to point_metadata.csv."
    )


# Create simple point_index -> Station lookup
station_by_index = matched.set_index(
    "point_index"
)["Station"].to_dict()


print("\nMatched stations:")

for point_index in sorted(station_by_index):
    print(
        f"  point {point_index}: "
        f"{station_by_index[point_index]}"
    )


# ============================================================
# 4. FIND ALL TIFF FILES
# ============================================================

tif_files = sorted(
    list(RASTER_FOLDER.glob("*.tif")) +
    list(RASTER_FOLDER.glob("*.tiff"))
)

if not tif_files:
    raise FileNotFoundError(
        f"No .tif or .tiff files found in: "
        f"{RASTER_FOLDER.resolve()}"
    )

print(f"\nFound {len(tif_files)} raster files.")


# ============================================================
# 5. CONTAINER FOR ALL RESULTS
# ============================================================

all_data = []


# ============================================================
# 6. PROCESS EACH RASTER
# ============================================================

for tif_path in tif_files:

    print("\n" + "=" * 60)
    print(f"Processing: {tif_path.name}")

    # --------------------------------------------------------
    # Extract date from filename
    # Expected example:
    # xxxx_20210425Txxxx.tif
    # --------------------------------------------------------

    match = re.search(r"(\d{8})T", tif_path.name)

    if match is None:
        print(
            "WARNING: No YYYYMMDDT date found in filename. "
            "Raster skipped."
        )
        continue

    date_str = match.group(1)

    try:
        date_obj = datetime.strptime(
            date_str,
            "%Y%m%d"
        )

    except ValueError:
        print(
            f"WARNING: Invalid date '{date_str}'. "
            "Raster skipped."
        )
        continue

    formatted_date = date_obj.strftime("%d.%m.%Y")


    # --------------------------------------------------------
    # Open raster
    # --------------------------------------------------------

    with rasterio.open(tif_path) as raster:

        print(f"Raster CRS: {raster.crs}")
        print(
            f"Raster resolution: "
            f"{abs(raster.res[0]):.2f} x "
            f"{abs(raster.res[1]):.2f}"
        )
        print(f"Raster NoData value: {raster.nodata}")

        if raster.crs is None:
            print(
                "WARNING: Raster has no CRS. "
                "Raster skipped."
            )
            continue


        # ----------------------------------------------------
        # Transform field points into raster CRS
        # ----------------------------------------------------

        if points_gdf.crs != raster.crs:
            points_raster_crs = points_gdf.to_crs(
                raster.crs
            )
        else:
            points_raster_crs = points_gdf.copy()


        # ----------------------------------------------------
        # Extract value at EVERY field-site coordinate
        # ----------------------------------------------------

        for idx, row in points_raster_crs.iterrows():

            point = row.geometry

            station = station_by_index.get(idx)

            if station is None:
                print(
                    f"WARNING: No station ID for point "
                    f"index {idx}. Skipping."
                )
                continue


            # -----------------------------------------------
            # Check whether point falls inside raster extent
            # -----------------------------------------------

            inside_raster = (
                raster.bounds.left <= point.x <= raster.bounds.right
                and
                raster.bounds.bottom <= point.y <= raster.bounds.top
            )

            if not inside_raster:

                print(
                    f"WARNING: {station} lies outside "
                    f"the raster extent."
                )

                raw_value = np.nan
                lai = np.nan
                raster_row = np.nan
                raster_col = np.nan

            else:

                # -------------------------------------------
                # Identify raster row/column containing point
                # -------------------------------------------

                raster_row, raster_col = raster.index(
                    point.x,
                    point.y
                )


                # -------------------------------------------
                # Sample raster pixel containing coordinate
                # -------------------------------------------

                sampled = next(
                    raster.sample(
                        [(point.x, point.y)],
                        indexes=1,
                        masked=True
                    )
                )

                pixel_value = sampled[0]


                # -------------------------------------------
                # Handle masked / NoData values
                # -------------------------------------------

                if np.ma.is_masked(pixel_value):

                    raw_value = np.nan
                    lai = np.nan

                else:

                    raw_value = float(pixel_value)

                    if (
                        raster.nodata is not None
                        and np.isclose(
                            raw_value,
                            raster.nodata
                        )
                    ):

                        lai = np.nan

                    else:

                        lai = (
                            raw_value *
                            LAI_SCALE_FACTOR
                        )


            # -----------------------------------------------
            # Round final LAI
            # -----------------------------------------------

            if np.isfinite(lai):
                lai_output = round(
                    lai,
                    LAI_DECIMALS
                )
            else:
                lai_output = np.nan


            # -----------------------------------------------
            # Store result
            # -----------------------------------------------

            all_data.append({

                "Point_ID": station,

                # ISO date is preferable internally
                "Date": date_obj.date(),

                # Human-readable version if useful
                "Date_formatted": formatted_date,

                "LAI": lai_output,

                "Raw_raster_value": raw_value,

                "Raster_row": raster_row,
                "Raster_col": raster_col,

                "Raster_file": tif_path.name,

                "Raster_resolution_x_m":
                    abs(raster.res[0]),

                "Raster_resolution_y_m":
                    abs(raster.res[1]),

                "Scale_factor":
                    LAI_SCALE_FACTOR
            })


            print(
                f"{station:4s} | "
                f"{formatted_date} | "
                f"row={raster_row}, "
                f"col={raster_col} | "
                f"LAI={lai_output}"
            )


# ============================================================
# 7. CREATE FINAL DATAFRAME
# ============================================================

combined_df = pd.DataFrame(all_data)

if combined_df.empty:
    raise ValueError(
        "No valid observations were extracted."
    )

combined_df = combined_df.sort_values(
    by=["Point_ID", "Date"]
).reset_index(drop=True)


# ============================================================
# 8. CREATE WIDE-FORMAT TABLE
# ============================================================

# Rows = dates
# Columns = stations
#
# Useful for visually checking temporal LAI patterns.

lai_wide = combined_df.pivot_table(
    index="Date",
    columns="Point_ID",
    values="LAI",
    aggfunc="first"
).reset_index()


# ============================================================
# 9. EXPORT CSV
# ============================================================

csv_path = OUTPUT_FOLDER / "combined_lai.csv"

combined_df.to_csv(
    csv_path,
    index=False
)

print(f"\nCSV saved to:")
print(csv_path.resolve())


# ============================================================
# 10. EXPORT EXCEL
# ============================================================

excel_path = OUTPUT_FOLDER / "combined_lai.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    # Long-format master dataset
    combined_df.to_excel(
        writer,
        sheet_name="LAI_long",
        index=False
    )

    # Convenient site × date representation
    lai_wide.to_excel(
        writer,
        sheet_name="LAI_by_site",
        index=False
    )


print("\nExcel workbook saved to:")
print(excel_path.resolve())

print("\nExtraction completed successfully.")