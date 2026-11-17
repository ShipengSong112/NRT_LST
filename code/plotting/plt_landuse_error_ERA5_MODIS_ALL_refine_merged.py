# -*- coding: utf-8 -*-
"""
Refined land-cover-stratified comparison of AMSR2-retrieved LST with
MODIS and ERA5 skin temperature.

Compared with plt_landuse_error_ERA5_MODIS_ALL.py, this version:
1. Aggregates the 500-m IGBP map to the 10-km MODIS/AMSR2 analysis
   footprint and retains only footprints with >= 80% dominant-class purity.
2. Uses the pre-aggregated 10-km MODIS footprint-mean LST and requires
   >= 80% valid MODIS subpixel coverage.
3. Uses the MODIS view-time bands for temporal quality control.
4. Interpolates ERA5 by its actual datetime coordinate and uses the
   representative observation time of the valid AMSR2 scan lines.
5. Selects dates at 10-day intervals from 1 January 2020.
6. Uses common sample support so the AMSR2-MODIS and AMSR2-ERA5 comparisons
   are based on the same pure, high-coverage MODIS footprints.
7. Merges evergreen and deciduous broadleaf forests, and merges woody
   savannas and savannas, before calculating 10-km land-cover purity.

The original script is intentionally left unchanged.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import rioxarray
import xarray as xr


plt.rcParams["font.sans-serif"] = ["Arial", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ================= Configuration =================

# --- 1. Paths ---
AMSR2_DIR_DAY = r"G:\E_LST\result\Surface_Retrieval_Robust_ROI_9channel_lst_Eprior_TELSEM\A"
AMSR2_DIR_NIGHT = r"G:\E_LST\result\Surface_Retrieval_Robust_ROI_9channel_lst_Eprior_TELSEM\D"

ERA5_ROOT_DIR = r"I:\ERA5\LST"
MODIS_DIR = r"G:\LST\MYD11A1\MODIS_LST_10km_qc_coverage1"
LANDUSE_PATH = r"I:\landsue\MODIS_landuse_2020.tif"

OUT_DIR = r"F:\E_LST\JPG\LST\landuse_TELSEM"
os.makedirs(OUT_DIR, exist_ok=True)
SAVE_PREFIX = os.path.join(
    OUT_DIR, "LST_Comparison_AMSR2_ERA5_MODIS_refine_merged"
)

# --- 2. Annual sampling ---
ANALYSIS_YEAR = 2020

# Sample continuously through the year from 1 January at this interval.
SAMPLE_INTERVAL_DAYS = 10

# --- 3. Matching and quality control ---
LAND_COVER_PURITY_THRESHOLD = 0.8
LAND_COVER_VALID_FRACTION_THRESHOLD = 0.8
MODIS_COVERAGE_THRESHOLD = 0.8
MAX_MODIS_TIME_DIFF_HOURS = 0.50

# The 10-km MODIS files already contain the mean of valid fine-resolution
# MODIS subpixels and their valid coverage fraction. Each matched regular-grid
# cell is therefore treated as the MODIS footprint mean.
MODIS_SCALE_FACTOR = 1.0

# AMSR2 rotates at approximately 40 rpm; one scan line is about 1.5 s.
# This is used to reconstruct scan-line observation time from granule time.
AMSR2_SCAN_INTERVAL_SECONDS = 1.5

# True: AMSR2-MODIS and AMSR2-ERA5 use identical samples that pass
# land-cover purity, MODIS coverage, and MODIS time matching.
# False: AMSR2-ERA5 can additionally use pure pixels without valid MODIS.
USE_COMMON_SAMPLE_SUPPORT = True

ERA5_SPATIAL_INTERP_METHOD = "linear"
MAX_BOXPLOT_POINTS_PER_GROUP = 100_000
RANDOM_SEED = 20200810

# Land-cover aggregation is cached because reading the 500-m China raster is
# the most expensive setup step.
LAND_COVER_CACHE = os.path.join(
    OUT_DIR, "MODIS_landuse_2020_to_MODIS10km_merged_mode_purity.npz"
)


# --- 4. IGBP classes ---
IGBP_DICT_EN = {
    1: "Evergreen Needleleaf",
    2: "Evergreen Broadleaf",
    3: "Deciduous Needleleaf",
    4: "Deciduous Broadleaf",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban/Built-up",
    14: "Cropland/Natural Mosaic",
    15: "Snow and Ice",
    16: "Barren/Sparsely Vegetated",
    17: "Water Bodies",
    18: "Broadleaf Forests",
    19: "Savannas",
}

VALID_IGBP_MIN = 1
VALID_IGBP_MAX = 17

# Codes 18 and 19 are analysis-group codes, not native IGBP codes.
# Purity is calculated after combining the corresponding 500-m classes.
LAND_COVER_GROUPS = {
    1: (1,),
    3: (3,),
    5: (5,),
    6: (6,),
    7: (7,),
    10: (10,),
    11: (11,),
    12: (12,),
    13: (13,),
    14: (14,),
    15: (15,),
    16: (16,),
    17: (17,),
    18: (2, 4),   # Evergreen + deciduous broadleaf forests
    19: (8, 9),   # Woody savannas + savannas
}

INTEREST_TYPES = [18, 5, 7, 19, 10, 12, 13, 15, 16]


# ================= General utilities =================

def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def parse_amsr2_start_datetime(filename):
    """Parse UTC granule start datetime from an AMSR2 result filename."""
    match = re.search(r"_(\d{8})(\d{4})_", os.path.basename(filename))
    if not match:
        raise ValueError(f"Cannot parse AMSR2 datetime: {filename}")
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


def filter_amsr2_files(file_list, target_date):
    """Filter AMSR2 NetCDF files by YYYYMMDD."""
    return sorted(
        f
        for f in file_list
        if f.lower().endswith(".nc") and f"_{target_date}" in os.path.basename(f)
    )


def get_era5_file(date_str):
    return os.path.join(ERA5_ROOT_DIR, date_str[:4], f"ERA5_{date_str}_land.nc")


def get_modis_file(date_str):
    return os.path.join(MODIS_DIR, f"MYD11A1_10km_{date_str}.tif")


def extract_dates_from_amsr2(directory, year):
    dates = set()
    for path in glob.glob(os.path.join(directory, "*.nc")):
        match = re.search(r"_(\d{8})\d{4}_", os.path.basename(path))
        if match and match.group(1).startswith(str(year)):
            dates.add(match.group(1))
    return dates


def extract_dates_from_modis(directory, year):
    dates = set()
    pattern = os.path.join(directory, f"MYD11A1_10km_{year}*.tif")
    for path in glob.glob(pattern):
        match = re.search(r"MYD11A1_10km_(\d{8})\.tif$", os.path.basename(path))
        if match:
            dates.add(match.group(1))
    return dates


def extract_dates_from_era5(root_dir, year):
    dates = set()
    pattern = os.path.join(root_dir, str(year), f"ERA5_{year}*_land.nc")
    for path in glob.glob(pattern):
        match = re.search(r"ERA5_(\d{8})_land\.nc$", os.path.basename(path))
        if match:
            dates.add(match.group(1))
    return dates


def choose_balanced_dates(common_dates, interval_days):
    """Select common dates every ``interval_days`` from 1 January."""
    common_dates = set(common_dates)
    current_date = datetime(ANALYSIS_YEAR, 1, 1)
    end_date = datetime(ANALYSIS_YEAR, 12, 31)
    selected = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        if date_str in common_dates:
            selected.append(date_str)
        current_date += timedelta(days=interval_days)
    return selected


def discover_target_dates():
    """Find dates available in MODIS, ERA5, and both AMSR2 orbit folders."""
    day_dates = extract_dates_from_amsr2(AMSR2_DIR_DAY, ANALYSIS_YEAR)
    night_dates = extract_dates_from_amsr2(AMSR2_DIR_NIGHT, ANALYSIS_YEAR)
    modis_dates = extract_dates_from_modis(MODIS_DIR, ANALYSIS_YEAR)
    era5_dates = extract_dates_from_era5(ERA5_ROOT_DIR, ANALYSIS_YEAR)

    common = day_dates & night_dates & modis_dates & era5_dates
    selected = choose_balanced_dates(common, SAMPLE_INTERVAL_DAYS)

    print(
        "Available dates - "
        f"AMSR2 day: {len(day_dates)}, AMSR2 night: {len(night_dates)}, "
        f"MODIS: {len(modis_dates)}, ERA5: {len(era5_dates)}, common: {len(common)}"
    )
    print(f"Selected {len(selected)} dates: {', '.join(selected)}")

    if not selected:
        raise RuntimeError("No common dates were found for the requested year.")

    return selected


# ================= Land-cover footprint aggregation =================

def file_signature(*paths):
    parts = []
    for path in paths:
        stat = os.stat(path)
        parts.append(f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}")
    return "||".join(parts)


def load_cached_land_cover(cache_path, expected_signature):
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != expected_signature:
                return None
            return (
                cache["dominant_class"],
                cache["purity"],
                cache["valid_fraction"],
            )
    except Exception:
        return None


def _read_padded_source_block(src, row_start, col_start, height, width):
    """Read a possibly out-of-bounds raster block, padding with zero."""
    block = np.zeros((height, width), dtype=np.uint8)

    src_row0 = max(0, row_start)
    src_col0 = max(0, col_start)
    src_row1 = min(src.height, row_start + height)
    src_col1 = min(src.width, col_start + width)

    if src_row1 <= src_row0 or src_col1 <= src_col0:
        return block

    data = src.read(
        1,
        window=Window(
            col_off=src_col0,
            row_off=src_row0,
            width=src_col1 - src_col0,
            height=src_row1 - src_row0,
        ),
    )

    dst_row0 = src_row0 - row_start
    dst_col0 = src_col0 - col_start
    block[
        dst_row0 : dst_row0 + data.shape[0],
        dst_col0 : dst_col0 + data.shape[1],
    ] = data
    return block


def aggregate_land_cover_to_modis_grid(landuse_path, template_path, cache_path):
    """
    Aggregate 500-m IGBP to the regular 10-km MODIS grid.

    For every 10-km footprint, return:
      - dominant land-cover group after the requested class merges;
      - fraction occupied by the dominant group;
      - fraction covered by valid IGBP pixels.

    The current files have exactly 20 x 20 land-cover pixels per MODIS cell.
    The implementation verifies this alignment rather than assuming it.
    """
    signature = (
        file_signature(landuse_path, template_path)
        + f"||groups={LAND_COVER_GROUPS!r}"
    )
    cached = load_cached_land_cover(cache_path, signature)
    if cached is not None:
        print(f"Loaded cached land-cover footprint purity: {cache_path}")
        return cached

    print("Aggregating 500-m land cover to 10-km footprints...")

    with rasterio.open(template_path) as template, rasterio.open(landuse_path) as src:
        if template.crs != src.crs:
            raise ValueError(
                f"CRS mismatch: MODIS={template.crs}, land cover={src.crs}. "
                "Reproject land cover before aggregation."
            )
        if template.crs is None or template.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326 analysis grid, got {template.crs}")

        src_xres, src_yres = map(abs, src.res)
        dst_xres, dst_yres = map(abs, template.res)
        ratio_x = int(round(dst_xres / src_xres))
        ratio_y = int(round(dst_yres / src_yres))

        if ratio_x <= 0 or ratio_y <= 0:
            raise ValueError("Invalid source/target resolution ratio.")
        if not np.isclose(dst_xres, ratio_x * src_xres, rtol=0, atol=1e-10):
            raise ValueError("Land-cover and MODIS x resolutions are not integer aligned.")
        if not np.isclose(dst_yres, ratio_y * src_yres, rtol=0, atol=1e-10):
            raise ValueError("Land-cover and MODIS y resolutions are not integer aligned.")

        col_start_float = (template.bounds.left - src.bounds.left) / src_xres
        row_start_float = (src.bounds.top - template.bounds.top) / src_yres
        col_start = int(round(col_start_float))
        row_start = int(round(row_start_float))

        if not np.isclose(col_start_float, col_start, atol=1e-5):
            raise ValueError("Land-cover and MODIS grid origins are not x-aligned.")
        if not np.isclose(row_start_float, row_start, atol=1e-5):
            raise ValueError("Land-cover and MODIS grid origins are not y-aligned.")

        height, width = template.height, template.width
        dominant = np.zeros((height, width), dtype=np.uint8)
        purity = np.full((height, width), np.nan, dtype=np.float32)
        valid_fraction = np.zeros((height, width), dtype=np.float32)

        # Chunked processing keeps memory stable for the 500-m China raster.
        chunk_target_rows = 16
        footprint_pixel_count = ratio_x * ratio_y

        for target_row0 in range(0, height, chunk_target_rows):
            target_row1 = min(height, target_row0 + chunk_target_rows)
            n_target_rows = target_row1 - target_row0

            source_row0 = row_start + target_row0 * ratio_y
            source_col0 = col_start
            source_height = n_target_rows * ratio_y
            source_width = width * ratio_x

            source_block = _read_padded_source_block(
                src,
                row_start=source_row0,
                col_start=source_col0,
                height=source_height,
                width=source_width,
            )

            pixels = (
                source_block.reshape(n_target_rows, ratio_y, width, ratio_x)
                .transpose(0, 2, 1, 3)
                .reshape(n_target_rows, width, footprint_pixel_count)
            )

            valid = (pixels >= VALID_IGBP_MIN) & (pixels <= VALID_IGBP_MAX)
            valid_count = valid.sum(axis=2, dtype=np.uint16)

            group_codes = np.asarray(list(LAND_COVER_GROUPS), dtype=np.uint8)
            counts = np.empty(
                (len(group_codes), n_target_rows, width), dtype=np.uint16
            )
            for group_index, source_codes in enumerate(
                LAND_COVER_GROUPS.values()
            ):
                counts[group_index] = np.isin(pixels, source_codes).sum(
                    axis=2, dtype=np.uint16
                )

            dominant_index = counts.argmax(axis=0)
            chunk_dominant = group_codes[dominant_index]
            chunk_dominant_count = counts.max(axis=0)
            no_valid = valid_count == 0
            chunk_dominant[no_valid] = 0

            chunk_purity = np.divide(
                chunk_dominant_count,
                valid_count,
                out=np.full(valid_count.shape, np.nan, dtype=np.float32),
                where=~no_valid,
            )

            dominant[target_row0:target_row1] = chunk_dominant
            purity[target_row0:target_row1] = chunk_purity
            valid_fraction[target_row0:target_row1] = (
                valid_count.astype(np.float32) / footprint_pixel_count
            )

            print(
                f"  land-cover rows {target_row0 + 1}-{target_row1}/{height}",
                end="\r",
            )

    print()
    np.savez_compressed(
        cache_path,
        signature=np.array(signature),
        dominant_class=dominant,
        purity=purity,
        valid_fraction=valid_fraction,
    )
    print(f"Saved land-cover footprint cache: {cache_path}")
    return dominant, purity, valid_fraction


def regular_grid_coordinates(raster_path):
    with rasterio.open(raster_path) as src:
        transform = src.transform
        x = transform.c + (np.arange(src.width) + 0.5) * transform.a
        y = transform.f + (np.arange(src.height) + 0.5) * transform.e
    return x, y


def build_land_cover_dataarrays(template_path, dominant, purity, valid_fraction):
    x, y = regular_grid_coordinates(template_path)
    coords = {"y": y, "x": x}
    return (
        xr.DataArray(dominant, dims=("y", "x"), coords=coords),
        xr.DataArray(purity, dims=("y", "x"), coords=coords),
        xr.DataArray(valid_fraction, dims=("y", "x"), coords=coords),
    )


# ================= Time and reference matching =================

def format_era5_longitude(era5_ds, lon_name):
    if float(era5_ds[lon_name].max()) > 180.0:
        era5_ds = era5_ds.assign_coords(
            {lon_name: (era5_ds[lon_name] + 180.0) % 360.0 - 180.0}
        )
        era5_ds = era5_ds.sortby(lon_name)
    return era5_ds


def get_era5_at_datetime(era5_ds, target_datetime):
    """Interpolate ERA5 skin temperature using its actual time coordinate."""
    time_name = "valid_time" if "valid_time" in era5_ds.coords else "time"
    if time_name not in era5_ds.coords:
        raise KeyError("ERA5 dataset has neither valid_time nor time coordinate.")

    target_time = np.datetime64(target_datetime, "ns")
    times = era5_ds[time_name].values.astype("datetime64[ns]")
    # ERA5 GRIB conversion may attach expver(valid_time) as an auxiliary
    # string coordinate. Some xarray versions cannot interpolate a scalar
    # datetime while that auxiliary coordinate is present, so retain only
    # dimension coordinates before interpolation.
    skt = era5_ds["skt"].reset_coords(drop=True)

    if target_time < times.min() or target_time > times.max():
        nearest_idx = int(np.argmin(np.abs(times - target_time)))
        result = skt.isel({time_name: nearest_idx}).squeeze(drop=True)
    else:
        result = skt.interp({time_name: target_time}).squeeze(drop=True)

    # xarray retains the interpolated datetime as a scalar coordinate. When
    # vectorized 2-D latitude/longitude targets are used later, that scalar
    # coordinate can conflict with the target dimensions. It is metadata only
    # at this stage, so remove it before spatial interpolation.
    if time_name in result.coords and time_name not in result.dims:
        result = result.reset_coords(time_name, drop=True)
    return result


def reconstruct_scanline_times(start_datetime, n_rows):
    seconds = np.arange(n_rows, dtype=np.float64) * AMSR2_SCAN_INTERVAL_SECONDS
    return np.array(
        [start_datetime + timedelta(seconds=float(v)) for v in seconds],
        dtype="datetime64[ns]",
    )


def representative_swath_datetime(start_datetime, valid_mask):
    """Return the median observation time of scan lines containing valid LST."""
    valid_rows = np.flatnonzero(np.any(valid_mask, axis=1))
    if valid_rows.size == 0:
        return start_datetime
    median_row = float(np.median(valid_rows))
    return start_datetime + timedelta(
        seconds=median_row * AMSR2_SCAN_INTERVAL_SECONDS
    )


def local_solar_time_grid(start_datetime, lon_swath):
    """Approximate local solar observation time for every AMSR2 pixel."""
    row_hours = (
        start_datetime.hour
        + start_datetime.minute / 60.0
        + start_datetime.second / 3600.0
        + np.arange(lon_swath.shape[0], dtype=np.float64)
        * AMSR2_SCAN_INTERVAL_SECONDS
        / 3600.0
    )
    return (row_hours[:, None] + lon_swath / 15.0) % 24.0


def circular_hour_difference(time_a, time_b):
    return np.abs((time_a - time_b + 12.0) % 24.0 - 12.0)


def sample_to_swath(data_array, lat_da, lon_da, method="nearest"):
    return data_array.interp(y=lat_da, x=lon_da, method=method).values


# ================= Statistics and plotting =================

def new_bias_collection():
    return {
        "Day": {code: [] for code in INTEREST_TYPES},
        "Night": {code: [] for code in INTEREST_TYPES},
    }


def append_grouped_errors(collection, orbit_mode, land_codes, mask, errors):
    for code in INTEREST_TYPES:
        selected = mask & (land_codes == code)
        if np.any(selected):
            collection[orbit_mode][code].append(
                errors[selected].astype(np.float32, copy=False)
            )


def combine_chunks(chunks):
    if not chunks:
        return np.empty(0, dtype=np.float32)
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def save_bias_stats_to_csv(bias_collection, csv_path, n_dates):
    print(f"Saving statistical metrics: {csv_path}")
    stats_list = []

    for orbit_mode in ["Day", "Night"]:
        for code in INTEREST_TYPES:
            arr = combine_chunks(bias_collection[orbit_mode].get(code, []))
            if arr.size == 0:
                continue

            mean_bias = float(np.mean(arr, dtype=np.float64))
            rmse = float(
                np.sqrt(np.mean(arr.astype(np.float64) ** 2, dtype=np.float64))
            )
            ubrmse = float(np.sqrt(max(0.0, rmse**2 - mean_bias**2)))

            stats_list.append(
                {
                    "Orbit_Mode": orbit_mode,
                    "IGBP_Code": code,
                    "LandCover_Type": IGBP_DICT_EN.get(code, str(code)),
                    "N_Samples": int(arr.size),
                    "N_Dates": int(n_dates),
                    "LandCover_Purity_Threshold": LAND_COVER_PURITY_THRESHOLD,
                    "MODIS_Coverage_Threshold": MODIS_COVERAGE_THRESHOLD,
                    "Mean_Bias (K)": round(mean_bias, 4),
                    "Median_Bias (K)": round(float(np.median(arr)), 4),
                    "MAE (K)": round(float(np.mean(np.abs(arr))), 4),
                    "RMSE (K)": round(rmse, 4),
                    "ubRMSE (K)": round(ubrmse, 4),
                    "Std_Dev (K)": round(float(np.std(arr)), 4),
                    "Min (K)": round(float(np.min(arr)), 4),
                    "Q1_25% (K)": round(float(np.percentile(arr, 25)), 4),
                    "Q3_75% (K)": round(float(np.percentile(arr, 75)), 4),
                    "Max (K)": round(float(np.max(arr)), 4),
                }
            )

    pd.DataFrame(stats_list).to_csv(csv_path, index=False)


def _plot_sample(arr, rng):
    arr = np.asarray(arr)
    if arr.size <= MAX_BOXPLOT_POINTS_PER_GROUP:
        return arr
    idx = rng.choice(arr.size, size=MAX_BOXPLOT_POINTS_PER_GROUP, replace=False)
    return arr[idx]


def plot_two_grouped_boxplots(bias_modis, bias_era5, save_path):
    print("Plotting two-panel land-cover error distributions...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 13.5), sharex=True, dpi=300
    )
    plt.subplots_adjust(hspace=0.08)
    rng = np.random.default_rng(RANDOM_SEED)

    def draw_box(ax, bias_coll, ylabel, panel_label):
        box_data, positions, colors, labels = [], [], [], []

        for current_pos, code in enumerate(INTEREST_TYPES, start=1):
            full_day = combine_chunks(bias_coll["Day"].get(code, []))
            full_night = combine_chunks(bias_coll["Night"].get(code, []))
            labels.append(IGBP_DICT_EN.get(code, str(code)))

            if full_day.size > 5:
                box_data.append(_plot_sample(full_day, rng))
                positions.append(current_pos - 0.2)
                colors.append("#FF9999")
            if full_night.size > 5:
                box_data.append(_plot_sample(full_night, rng))
                positions.append(current_pos + 0.2)
                colors.append("#66B2FF")

        if box_data:
            box = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.35,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_edgecolor("#444444")
            for median in box["medians"]:
                median.set(color="#C0392B", linewidth=2)
            for mean in box["means"]:
                mean.set(
                    marker="D",
                    markeredgecolor="black",
                    markerfacecolor="white",
                    markersize=5,
                )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.text(
            0.015,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="top",
        )
        return labels

    draw_box(
        ax1,
        bias_modis,
        "LST difference: AMSR2 − MODIS (K)",
        "(a)",
    )
    labels = draw_box(
        ax2,
        bias_era5,
        "LST difference: AMSR2 − ERA5-Land (K)",
        "(b)",
    )

    ax2.set_xticks(range(1, len(INTEREST_TYPES) + 1))
    ax2.set_xticklabels(
        labels,
        fontsize=11,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax2.set_xlabel("")

    legend_elements = [
        mpatches.Patch(
            facecolor="#FF9999", edgecolor="#444444", label="Day"
        ),
        mpatches.Patch(
            facecolor="#66B2FF", edgecolor="#444444", label="Night"
        ),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=2,
        fontsize=14,
        handlelength=2.0,
        handleheight=1.2,
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot: {save_path}")


# ================= Main processing =================

def main():
    require_path(AMSR2_DIR_DAY, "AMSR2 daytime directory")
    require_path(AMSR2_DIR_NIGHT, "AMSR2 nighttime directory")
    require_path(ERA5_ROOT_DIR, "ERA5 root directory")
    require_path(MODIS_DIR, "MODIS directory")
    require_path(LANDUSE_PATH, "500-m land-cover raster")

    target_dates = discover_target_dates()

    template_path = get_modis_file(target_dates[0])
    require_path(template_path, "MODIS template")

    dominant, purity, valid_fraction = aggregate_land_cover_to_modis_grid(
        LANDUSE_PATH, template_path, LAND_COVER_CACHE
    )
    land_class_da, land_purity_da, land_valid_da = build_land_cover_dataarrays(
        template_path, dominant, purity, valid_fraction
    )

    bias_collection_era5 = new_bias_collection()
    bias_collection_modis = new_bias_collection()

    runs = [
        ("Day", AMSR2_DIR_DAY, 1, 2, 3),
        ("Night", AMSR2_DIR_NIGHT, 4, 5, 6),
    ]

    for orbit_mode, amsr2_dir, lst_band, time_band, coverage_band in runs:
        print(f"\n========== Processing {orbit_mode} ==========")
        amsr2_files_all = glob.glob(os.path.join(amsr2_dir, "*.nc"))

        for date_index, date_str in enumerate(target_dates, start=1):
            print(
                f"[{orbit_mode}] {date_index}/{len(target_dates)} - {date_str}"
            )

            era5_path = get_era5_file(date_str)
            modis_path = get_modis_file(date_str)
            if not os.path.exists(era5_path) or not os.path.exists(modis_path):
                print("  Missing ERA5 or MODIS; skipped.")
                continue

            era5_ds = xr.open_dataset(era5_path)
            lat_name = "latitude" if "latitude" in era5_ds.coords else "lat"
            lon_name = "longitude" if "longitude" in era5_ds.coords else "lon"
            era5_ds = format_era5_longitude(era5_ds, lon_name)

            modis_stack = rioxarray.open_rasterio(modis_path, masked=True)
            if modis_stack.sizes.get("band", 0) < 6:
                modis_stack.close()
                era5_ds.close()
                raise ValueError(f"Expected six MODIS bands: {modis_path}")

            modis_lst_da = modis_stack.isel(band=lst_band - 1) * MODIS_SCALE_FACTOR
            modis_time_da = modis_stack.isel(band=time_band - 1)
            modis_coverage_da = modis_stack.isel(band=coverage_band - 1)

            amsr2_files = filter_amsr2_files(amsr2_files_all, date_str)

            for f_path in amsr2_files:
                try:
                    start_datetime = parse_amsr2_start_datetime(f_path)

                    with xr.open_dataset(f_path) as ds:
                        lst_swath = ds["Retrieved_T_skin_K"].values
                        #lst_swath = ds["Retrieved_T_skin(K)"].values
                        lat_swath = ds["lat"].values
                        lon_swath = ds["lon"].values

                    base_valid = (
                        np.isfinite(lst_swath)
                        & np.isfinite(lat_swath)
                        & np.isfinite(lon_swath)
                    )
                    if not np.any(base_valid):
                        continue

                    lat_da = xr.DataArray(lat_swath, dims=("y", "x"))
                    lon_da = xr.DataArray(lon_swath, dims=("y", "x"))

                    # Match the precomputed 10-km land-cover footprint mode/purity.
                    land_class_raw = sample_to_swath(
                        land_class_da, lat_da, lon_da, method="nearest"
                    )
                    land_purity = sample_to_swath(
                        land_purity_da, lat_da, lon_da, method="nearest"
                    )
                    land_valid_fraction = sample_to_swath(
                        land_valid_da, lat_da, lon_da, method="nearest"
                    )
                    land_codes = np.where(
                        np.isfinite(land_class_raw), np.rint(land_class_raw), 0
                    ).astype(np.uint8)

                    pure_land_mask = (
                        base_valid
                        & np.isfinite(land_purity)
                        & (land_purity >= LAND_COVER_PURITY_THRESHOLD)
                        & (land_valid_fraction >= LAND_COVER_VALID_FRACTION_THRESHOLD)
                        & np.isin(land_codes, INTEREST_TYPES)
                    )

                    # Match the existing 10-km MODIS footprint means and QC fields.
                    modis_lst = sample_to_swath(
                        modis_lst_da, lat_da, lon_da, method="nearest"
                    )
                    modis_view_time = sample_to_swath(
                        modis_time_da, lat_da, lon_da, method="nearest"
                    )
                    modis_coverage = sample_to_swath(
                        modis_coverage_da, lat_da, lon_da, method="nearest"
                    )

                    amsr_local_time = local_solar_time_grid(
                        start_datetime, lon_swath
                    )
                    time_difference = circular_hour_difference(
                        amsr_local_time, modis_view_time
                    )

                    modis_qc_mask = (
                        np.isfinite(modis_lst)
                        & np.isfinite(modis_view_time)
                        & np.isfinite(modis_coverage)
                        & (modis_coverage >= MODIS_COVERAGE_THRESHOLD)
                        & (time_difference <= MAX_MODIS_TIME_DIFF_HOURS)
                    )

                    # Use the representative time of valid China scan lines for ERA5.
                    target_datetime = representative_swath_datetime(
                        start_datetime, base_valid
                    )
                    era5_time_slice = get_era5_at_datetime(
                        era5_ds, target_datetime
                    )
                    era5_matched = era5_time_slice.interp(
                        {lat_name: lat_da, lon_name: lon_da},
                        method=ERA5_SPATIAL_INTERP_METHOD,
                    ).values
                    era5_valid = np.isfinite(era5_matched)

                    if USE_COMMON_SAMPLE_SUPPORT:
                        common_mask = pure_land_mask & modis_qc_mask & era5_valid
                        mask_modis = common_mask
                        mask_era5 = common_mask
                    else:
                        mask_modis = pure_land_mask & modis_qc_mask
                        mask_era5 = pure_land_mask & era5_valid

                    diff_modis = lst_swath - modis_lst
                    diff_era5 = lst_swath - era5_matched

                    append_grouped_errors(
                        bias_collection_modis,
                        orbit_mode,
                        land_codes,
                        mask_modis,
                        diff_modis,
                    )
                    append_grouped_errors(
                        bias_collection_era5,
                        orbit_mode,
                        land_codes,
                        mask_era5,
                        diff_era5,
                    )
                except Exception as exc:
                    print(f"  Error processing {os.path.basename(f_path)}: {exc}")

            modis_stack.close()
            era5_ds.close()

    plot_two_grouped_boxplots(
        bias_collection_modis,
        bias_collection_era5,
        f"{SAVE_PREFIX}_Boxplot_AMSR2_MODIS_ERA5.jpg",
    )

    save_bias_stats_to_csv(
        bias_collection_modis,
        f"{SAVE_PREFIX}_MODIS_Stats.csv",
        len(target_dates),
    )
    save_bias_stats_to_csv(
        bias_collection_era5,
        f"{SAVE_PREFIX}_ERA5_Stats.csv",
        len(target_dates),
    )
    print("\nRefined land-cover comparison completed.")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
