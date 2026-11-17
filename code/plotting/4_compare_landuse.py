# -*- coding: utf-8 -*-
"""
Compare four LST retrieval models by merged land-cover class.

The script uses common matched samples across all four models, MODIS, and
ERA5. It writes one metrics CSV and two bar-chart figures: Day and Night.
Each figure uses three metric rows (R/RMSE/Bias) and two reference columns
(MODIS/ERA5).
"""

import glob
import os
import textwrap
from contextlib import closing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

import plt_landuse_error_ERA5_MODIS_ALL_refine_merged as refine


# =============================================================================
# 0. Configuration
# =============================================================================
RETRIEVAL_ROOT = r"G:\E_LST\result\4_model_compare_dynamic_noise"
ERA5_ROOT_DIR = r"I:\ERA5\LST"
MODIS_DIR = r"G:\LST\MYD11A1\MODIS_LST_10km_qc_coverage1"
LANDUSE_PATH = r"I:\landsue\MODIS_landuse_2020.tif"
OUTPUT_DIR = r"F:\E_LST\JPG\LST\4_compare_landuse_dynamic_noise"

ANALYSIS_YEAR = 2020
SAMPLE_INTERVAL_DAYS = 10

MODELS = ["NN_BT", "NN_BT_Noise", "NN_BTE", "NN_BTE_Noise"]
PERIODS = {
    "Day": {"LST_BAND": 1, "TIME_BAND": 2, "COVERAGE_BAND": 3},
    "Night": {"LST_BAND": 4, "TIME_BAND": 5, "COVERAGE_BAND": 6},
}
REFERENCES = ["MODIS", "ERA5"]

MODEL_COLORS = {
    "NN_BT": "#4C78A8",
    "NN_BT_Noise": "#72B7B2",
    "NN_BTE": "#F58518",
    "NN_BTE_Noise": "#E45756",
}

INTEREST_TYPES = refine.INTEREST_TYPES
IGBP_DICT_EN = refine.IGBP_DICT_EN

LAND_COVER_PURITY_THRESHOLD = refine.LAND_COVER_PURITY_THRESHOLD
LAND_COVER_VALID_FRACTION_THRESHOLD = (
    refine.LAND_COVER_VALID_FRACTION_THRESHOLD
)
MODIS_COVERAGE_THRESHOLD = refine.MODIS_COVERAGE_THRESHOLD
MAX_MODIS_TIME_DIFF_HOURS = refine.MAX_MODIS_TIME_DIFF_HOURS
MODIS_SCALE_FACTOR = refine.MODIS_SCALE_FACTOR
ERA5_SPATIAL_INTERP_METHOD = refine.ERA5_SPATIAL_INTERP_METHOD

LAND_COVER_CACHE = os.path.join(
    OUTPUT_DIR,
    "MODIS_landuse_2020_to_MODIS10km_merged_mode_purity.npz",
)
METRICS_CSV = os.path.join(OUTPUT_DIR, "4_compare_landuse_metrics.csv")

BAR_WIDTH = 0.19

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
    }
)


# =============================================================================
# 1. Online paired statistics
# =============================================================================
class OnlinePairStatistics:
    """Accumulate exact R, RMSE, and bias without retaining all samples."""

    def __init__(self):
        self.n = 0
        self.sum_reference = 0.0
        self.sum_retrieval = 0.0
        self.sum_reference2 = 0.0
        self.sum_retrieval2 = 0.0
        self.sum_cross = 0.0
        self.sum_error = 0.0
        self.sum_error2 = 0.0

    def update(self, reference, retrieval):
        reference = np.asarray(reference, dtype=np.float64)
        retrieval = np.asarray(retrieval, dtype=np.float64)
        valid = np.isfinite(reference) & np.isfinite(retrieval)
        if not np.any(valid):
            return

        reference = reference[valid]
        retrieval = retrieval[valid]
        error = retrieval - reference

        self.n += int(reference.size)
        self.sum_reference += float(np.sum(reference, dtype=np.float64))
        self.sum_retrieval += float(np.sum(retrieval, dtype=np.float64))
        self.sum_reference2 += float(np.dot(reference, reference))
        self.sum_retrieval2 += float(np.dot(retrieval, retrieval))
        self.sum_cross += float(np.dot(reference, retrieval))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_error2 += float(np.dot(error, error))

    def results(self):
        if self.n == 0:
            return {"N": 0, "R": np.nan, "RMSE_K": np.nan, "Bias_K": np.nan}

        n = float(self.n)
        bias = self.sum_error / n
        rmse = np.sqrt(self.sum_error2 / n)

        sxx = self.sum_reference2 - self.sum_reference**2 / n
        syy = self.sum_retrieval2 - self.sum_retrieval**2 / n
        sxy = self.sum_cross - self.sum_reference * self.sum_retrieval / n
        if sxx > 0.0 and syy > 0.0:
            correlation = float(np.clip(sxy / np.sqrt(sxx * syy), -1.0, 1.0))
        else:
            correlation = np.nan

        return {
            "N": self.n,
            "R": correlation,
            "RMSE_K": float(rmse),
            "Bias_K": float(bias),
        }


def initialise_statistics():
    return {
        (model, period, code, reference): OnlinePairStatistics()
        for model in MODELS
        for period in PERIODS
        for code in INTEREST_TYPES
        for reference in REFERENCES
    }


# =============================================================================
# 2. Input discovery and loading
# =============================================================================
def retrieval_directory(model, period):
    return os.path.join(RETRIEVAL_ROOT, model, period)


def get_era5_file(date_str):
    return os.path.join(
        ERA5_ROOT_DIR, date_str[:4], f"ERA5_{date_str}_land.nc"
    )


def get_modis_file(date_str):
    return os.path.join(MODIS_DIR, f"MYD11A1_10km_{date_str}.tif")


def discover_target_dates():
    date_sets = []
    descriptions = []

    for model in MODELS:
        for period in PERIODS:
            directory = retrieval_directory(model, period)
            dates = refine.extract_dates_from_amsr2(directory, ANALYSIS_YEAR)
            date_sets.append(dates)
            descriptions.append(f"{model}/{period}: {len(dates)}")

    modis_dates = refine.extract_dates_from_modis(MODIS_DIR, ANALYSIS_YEAR)
    era5_dates = refine.extract_dates_from_era5(ERA5_ROOT_DIR, ANALYSIS_YEAR)
    date_sets.extend([modis_dates, era5_dates])
    descriptions.extend(
        [f"MODIS: {len(modis_dates)}", f"ERA5: {len(era5_dates)}"]
    )

    common_dates = set.intersection(*date_sets)
    selected = refine.choose_balanced_dates(
        common_dates, SAMPLE_INTERVAL_DAYS
    )

    print("Available dates - " + ", ".join(descriptions))
    print(f"Common dates: {len(common_dates)}")
    print(f"Selected {len(selected)} dates: {', '.join(selected)}")
    if not selected:
        raise RuntimeError("No common dates were found for all four models.")
    return selected


def common_granules(period, date_str):
    files_by_model = {}
    for model in MODELS:
        directory = retrieval_directory(model, period)
        paths = refine.filter_amsr2_files(
            glob.glob(os.path.join(directory, "*.nc")), date_str
        )
        files_by_model[model] = {
            os.path.basename(path): path for path in paths
        }

    common_names = set.intersection(
        *(set(files_by_model[model]) for model in MODELS)
    )
    return [
        (name, {model: files_by_model[model][name] for model in MODELS})
        for name in sorted(common_names)
    ]


def get_retrieved_lst(dataset):
    for variable_name in ("Retrieved_T_skin_K", "Retrieved_T_skin(K)"):
        if variable_name in dataset:
            return dataset[variable_name].values
    raise KeyError("No retrieved LST variable found in result file.")


def load_common_retrievals(paths_by_model):
    retrievals = {}
    lat = None
    lon = None

    for model in MODELS:
        with xr.open_dataset(paths_by_model[model]) as dataset:
            current_retrieval = get_retrieved_lst(dataset)
            current_lat = dataset["lat"].values
            current_lon = dataset["lon"].values

        if lat is None:
            lat = current_lat
            lon = current_lon
        else:
            if current_retrieval.shape != retrievals[MODELS[0]].shape:
                raise ValueError(f"Retrieval shape mismatch for {model}.")
            if current_lat.shape != lat.shape or current_lon.shape != lon.shape:
                raise ValueError(f"Coordinate shape mismatch for {model}.")
            if not np.allclose(current_lat, lat, rtol=0.0, atol=1.0e-6, equal_nan=True):
                raise ValueError(f"Latitude mismatch for {model}.")
            if not np.allclose(current_lon, lon, rtol=0.0, atol=1.0e-6, equal_nan=True):
                raise ValueError(f"Longitude mismatch for {model}.")

        retrievals[model] = current_retrieval

    return retrievals, lat, lon


# =============================================================================
# 3. Common-support matching and metric accumulation
# =============================================================================
def collect_statistics(target_dates, land_cover_arrays):
    land_class_da, land_purity_da, land_valid_da = land_cover_arrays
    statistics = initialise_statistics()

    for period, band_config in PERIODS.items():
        print(f"\n========== Processing {period} ==========")

        for date_index, date_str in enumerate(target_dates, start=1):
            print(f"[{period}] {date_index}/{len(target_dates)} - {date_str}")
            era5_path = get_era5_file(date_str)
            modis_path = get_modis_file(date_str)

            with xr.open_dataset(era5_path) as era5_source:
                lat_name = (
                    "latitude" if "latitude" in era5_source.coords else "lat"
                )
                lon_name = (
                    "longitude" if "longitude" in era5_source.coords else "lon"
                )
                era5_ds = refine.format_era5_longitude(
                    era5_source, lon_name
                )

                with closing(
                    rioxarray.open_rasterio(modis_path, masked=True)
                ) as modis_stack:
                    if modis_stack.sizes.get("band", 0) < 6:
                        raise ValueError(f"Expected six MODIS bands: {modis_path}")

                    modis_lst_da = (
                        modis_stack.isel(
                            band=band_config["LST_BAND"] - 1
                        )
                        * MODIS_SCALE_FACTOR
                    )
                    modis_time_da = modis_stack.isel(
                        band=band_config["TIME_BAND"] - 1
                    )
                    modis_coverage_da = modis_stack.isel(
                        band=band_config["COVERAGE_BAND"] - 1
                    )

                    granules = common_granules(period, date_str)
                    print(f"  Common granules: {len(granules)}")

                    for granule_name, paths_by_model in granules:
                        try:
                            retrievals, lat, lon = load_common_retrievals(
                                paths_by_model
                            )
                            common_retrieval_valid = np.logical_and.reduce(
                                [
                                    np.isfinite(retrievals[model])
                                    for model in MODELS
                                ]
                            )
                            common_retrieval_valid &= (
                                np.isfinite(lat) & np.isfinite(lon)
                            )
                            if not np.any(common_retrieval_valid):
                                continue

                            lat_da = xr.DataArray(lat, dims=("y", "x"))
                            lon_da = xr.DataArray(lon, dims=("y", "x"))

                            land_class_raw = refine.sample_to_swath(
                                land_class_da, lat_da, lon_da, method="nearest"
                            )
                            land_purity = refine.sample_to_swath(
                                land_purity_da, lat_da, lon_da, method="nearest"
                            )
                            land_valid_fraction = refine.sample_to_swath(
                                land_valid_da, lat_da, lon_da, method="nearest"
                            )
                            land_codes = np.where(
                                np.isfinite(land_class_raw),
                                np.rint(land_class_raw),
                                0,
                            ).astype(np.uint8)

                            pure_land_mask = (
                                common_retrieval_valid
                                & np.isfinite(land_purity)
                                & (
                                    land_purity
                                    >= LAND_COVER_PURITY_THRESHOLD
                                )
                                & (
                                    land_valid_fraction
                                    >= LAND_COVER_VALID_FRACTION_THRESHOLD
                                )
                                & np.isin(land_codes, INTEREST_TYPES)
                            )

                            modis_lst = refine.sample_to_swath(
                                modis_lst_da, lat_da, lon_da, method="nearest"
                            )
                            modis_view_time = refine.sample_to_swath(
                                modis_time_da, lat_da, lon_da, method="nearest"
                            )
                            modis_coverage = refine.sample_to_swath(
                                modis_coverage_da,
                                lat_da,
                                lon_da,
                                method="nearest",
                            )

                            start_datetime = refine.parse_amsr2_start_datetime(
                                granule_name
                            )
                            amsr_local_time = refine.local_solar_time_grid(
                                start_datetime, lon
                            )
                            time_difference = refine.circular_hour_difference(
                                amsr_local_time, modis_view_time
                            )
                            modis_qc_mask = (
                                np.isfinite(modis_lst)
                                & np.isfinite(modis_view_time)
                                & np.isfinite(modis_coverage)
                                & (
                                    modis_coverage
                                    >= MODIS_COVERAGE_THRESHOLD
                                )
                                & (
                                    time_difference
                                    <= MAX_MODIS_TIME_DIFF_HOURS
                                )
                            )

                            target_datetime = (
                                refine.representative_swath_datetime(
                                    start_datetime, common_retrieval_valid
                                )
                            )
                            era5_time_slice = refine.get_era5_at_datetime(
                                era5_ds, target_datetime
                            )
                            era5_matched = era5_time_slice.interp(
                                {lat_name: lat_da, lon_name: lon_da},
                                method=ERA5_SPATIAL_INTERP_METHOD,
                            ).values

                            common_mask = (
                                pure_land_mask
                                & modis_qc_mask
                                & np.isfinite(era5_matched)
                            )

                            for code in INTEREST_TYPES:
                                selected = common_mask & (land_codes == code)
                                if not np.any(selected):
                                    continue

                                reference_values = {
                                    "MODIS": modis_lst[selected],
                                    "ERA5": era5_matched[selected],
                                }
                                for model in MODELS:
                                    retrieval_values = retrievals[model][selected]
                                    for reference in REFERENCES:
                                        statistics[
                                            (model, period, code, reference)
                                        ].update(
                                            reference_values[reference],
                                            retrieval_values,
                                        )
                        except Exception as exc:
                            print(f"  Failed {granule_name}: {exc}")

    return statistics


# =============================================================================
# 4. Metrics table and bar charts
# =============================================================================
def build_metrics_table(statistics, n_dates):
    rows = []
    for period in PERIODS:
        for reference in REFERENCES:
            for code in INTEREST_TYPES:
                for model in MODELS:
                    result = statistics[
                        (model, period, code, reference)
                    ].results()
                    rows.append(
                        {
                            "Period": period,
                            "Reference": reference,
                            "Model": model,
                            "IGBP_Code": code,
                            "LandCover_Type": IGBP_DICT_EN[code],
                            "N_Samples": result["N"],
                            "N_Dates": n_dates,
                            "R": result["R"],
                            "RMSE (K)": result["RMSE_K"],
                            "Mean_Bias (K)": result["Bias_K"],
                            "LandCover_Purity_Threshold": (
                                LAND_COVER_PURITY_THRESHOLD
                            ),
                            "MODIS_Coverage_Threshold": (
                                MODIS_COVERAGE_THRESHOLD
                            ),
                        }
                    )

    metrics = pd.DataFrame(rows)

    for period in PERIODS:
        for reference in REFERENCES:
            for code in INTEREST_TYPES:
                subset = metrics.loc[
                    (metrics["Period"] == period)
                    & (metrics["Reference"] == reference)
                    & (metrics["IGBP_Code"] == code)
                ]
                counts = subset["N_Samples"].to_numpy()
                if np.any(counts == 0):
                    raise ValueError(
                        f"No common samples for {period}, {reference}, "
                        f"IGBP {code}."
                    )
                if len(np.unique(counts)) != 1:
                    raise ValueError(
                        f"Sample counts differ across models for {period}, "
                        f"{reference}, IGBP {code}: {counts.tolist()}"
                    )

    return metrics


def wrap_landcover_labels():
    return [
        textwrap.fill(
            IGBP_DICT_EN[code], width=14, break_long_words=False
        )
        for code in INTEREST_TYPES
    ]


def set_metric_limits(ax, metric, values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return

    if metric == "R":
        lower = min(0.0, float(finite.min()) - 0.05)
        upper = max(1.0, float(finite.max()) + 0.05)
        ax.set_ylim(max(-1.0, lower), min(1.05, upper))
    elif metric == "RMSE (K)":
        ax.set_ylim(0.0, float(finite.max()) * 1.15)
    else:
        max_abs = float(np.max(np.abs(finite)))
        padding = max_abs * 1.15 if max_abs > 0.0 else 1.0
        ax.set_ylim(-padding, padding)
        ax.axhline(0.0, color="black", linewidth=0.8)


def plot_period(metrics, period):
    metric_columns = ["R", "RMSE (K)", "Mean_Bias (K)"]
    metric_titles = ["R", "RMSE", "Bias"]
    x = np.arange(len(INTEREST_TYPES) + 1, dtype=np.float64)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18, 16),
        sharex=True,
        constrained_layout=True,
    )

    for row_index, (metric, title) in enumerate(
        zip(metric_columns, metric_titles)
    ):
        for column_index, reference in enumerate(REFERENCES):
            ax = axes[row_index, column_index]
            all_values = []

            for model_index, model in enumerate(MODELS):
                subset = (
                    metrics.loc[
                        (metrics["Period"] == period)
                        & (metrics["Reference"] == reference)
                        & (metrics["Model"] == model)
                    ]
                    .set_index("IGBP_Code")
                    .reindex(INTEREST_TYPES)
                )
                landcover_values = subset[metric].to_numpy(dtype=np.float64)
                # First bar: equal-weight mean across the nine land-cover classes.
                values = np.concatenate(
                    ([np.mean(landcover_values)], landcover_values)
                )
                all_values.extend(values)
                offset = (
                    model_index - (len(MODELS) - 1) / 2.0
                ) * BAR_WIDTH
                ax.bar(
                    x + offset,
                    values,
                    BAR_WIDTH,
                    label=model,
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.35,
                )

            set_metric_limits(ax, metric, all_values)
            ax.set_title(f"{reference}: {title}", fontweight="bold")
            ax.set_ylabel(title if title == "R" else f"{title} (K)")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

    labels = ["Mean"] + wrap_landcover_labels()
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("Land-cover type")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"{period}: four-model comparison by land-cover type",
        fontsize=16,
        fontweight="bold",
    )

    output_path = os.path.join(
        OUTPUT_DIR, f"4_compare_landuse_{period}.png"
    )
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# 5. Main
# =============================================================================
def main():
    for model in MODELS:
        for period in PERIODS:
            refine.require_path(
                retrieval_directory(model, period),
                f"{model} {period} retrieval directory",
            )
    refine.require_path(ERA5_ROOT_DIR, "ERA5 root directory")
    refine.require_path(MODIS_DIR, "MODIS directory")
    refine.require_path(LANDUSE_PATH, "500-m land-cover raster")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_dates = discover_target_dates()
    template_path = get_modis_file(target_dates[0])
    refine.require_path(template_path, "MODIS template")

    dominant, purity, valid_fraction = (
        refine.aggregate_land_cover_to_modis_grid(
            LANDUSE_PATH, template_path, LAND_COVER_CACHE
        )
    )
    land_cover_arrays = refine.build_land_cover_dataarrays(
        template_path, dominant, purity, valid_fraction
    )

    statistics = collect_statistics(target_dates, land_cover_arrays)
    metrics = build_metrics_table(statistics, len(target_dates))
    metrics.to_csv(METRICS_CSV, index=False, float_format="%.6f")
    print(f"Saved: {METRICS_CSV}")

    for period in PERIODS:
        plot_period(metrics, period)

    print("\nFour-model land-cover comparison completed.")


if __name__ == "__main__":
    main()
