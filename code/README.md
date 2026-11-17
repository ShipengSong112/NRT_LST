# Code overview

The scripts retain the local paths used in the original experiments. Update their configuration blocks before use.

| Script | Purpose |
|---|---|
| `training/4_model_compare_dynamic_noise.py` | Trains and tests `NN_BT`, `NN_BT_Noise`, `NN_BTE`, and `NN_BTE_Noise`. |
| `inference/4_model_compare_predict.py` | Applies the four trained models to 2020 AMSR2 observations. |
| `analysis/band_sensitive.py` | Calculates SHAP-based channel contributions. |
| `analysis/heihe_validation_lstonly.py` | Evaluates `NN_BT` against Daman, Huazhaizi, and Hunhelin observations. |
| `plotting/4_compare_landuse.py` | Compares the four algorithms with MODIS and ERA5-Land by land-cover class. |
| `plotting/plt_landuse_error_ERA5_MODIS_ALL_refine_merged.py` | Provides matching, aggregation, and metric functions for the land-cover comparison. |
| `plotting/plt_map_new.py` | Produces the daytime spatial comparison. |
| `plotting/plt_map_night.py` | Produces the nighttime spatial comparison. |

Large input datasets and model checkpoints are not included. The repository also does not contain every upstream script used to generate the RTTOV lookup table or preprocess the original datasets.
