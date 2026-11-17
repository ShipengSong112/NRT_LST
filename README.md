# NRT_LST

Code for **Near-Real-Time All-Weather LST Retrieval from AMSR2 by Combining Radiative Transfer and Deep Learning**.

The study compares four neural-network configurations:

- `NN_BT`: brightness temperature only;
- `NN_BT_Noise`: brightness temperature with noise-augmented training;
- `NN_BTE`: brightness temperature and a TELSEM2 emissivity prior;
- `NN_BTE_Noise`: brightness temperature and emissivity with noise-augmented training.

## Structure

```text
code/
├── training/    Model training and simulated-data evaluation
├── inference/   Retrieval from AMSR2 observations
├── analysis/    SHAP analysis and ground validation
└── plotting/    Land-cover and spatial comparisons
```

See [`code/README.md`](code/README.md) for the purpose of each script.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA version.

## Usage

1. Update the data and output paths in each script's configuration block.
2. Run `code/training/4_model_compare_dynamic_noise.py` to train and test the four configurations.
3. Run `code/inference/4_model_compare_predict.py` for AMSR2 retrieval.
4. Use the scripts in `code/analysis/` and `code/plotting/` for evaluation and figures.

Large datasets, model checkpoints, and retrieval products are not included. The repository contains the training, inference, and main evaluation code, but not every upstream step used to generate the RTTOV lookup table or preprocess the source datasets.
