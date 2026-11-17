# -*- coding: utf-8 -*-
"""
AMSR2 LST inference for four model modes, each with Day and Night checkpoints.

The selectable modes are:
    NN_BT, NN_BT_Noise, NN_BTE, NN_BTE_Noise

Noise is used only during noise-augmented model training. No synthetic BT or
emissivity noise is added during inference.
"""

import glob
import os
import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Inference running on device: {device}')

# =============================================================================
# 0. 全局配置
# =============================================================================
CONFIG = {
    'START_DATE': '20200101',
    'END_DATE': '20201231',
    'ROI': {
        'LAT_MIN': 15.0,
        'LAT_MAX': 55.0,
        'LON_MIN': 70.0,
        'LON_MAX': 135.0
    },
    'AMSR2_DIR': r'I:\AMSR2_G\2020',
    'TELSEM_DIR': r'I:\TELSEM\nc_amsr2',
    'MODEL_DIR': r'F:\E_LST\model',
    'RESULT_ROOT': r'G:\E_LST\result\4_model_compare_dynamic_noise',
    'INFERENCE_BATCH_SIZE': 20000
}

# 可填写一种或多种模式；默认运行全部4种模式。
SELECTED_MODES = ['NN_BT', 'NN_BT_Noise', 'NN_BTE', 'NN_BTE_Noise']

MODEL_MODES = {
    'NN_BT': {'USE_E': False},
    'NN_BT_Noise': {'USE_E': False},
    'NN_BTE': {'USE_E': True},
    'NN_BTE_Noise': {'USE_E': True}
}

# A轨对应白天模型，D轨对应黑夜模型。
PERIODS = {
    'Day': {'ORBIT': 'A', 'MODEL_SUFFIX': 'day_dynamic_noise'},
    'Night': {'ORBIT': 'D', 'MODEL_SUFFIX': 'night_dynamic_noise'}
}

BANDS_ORDER = [
    '10V', '10H', '18V', '18H', '23V',
    '23H', '36V', '36H', '89V', '89H'
]


# =============================================================================
# 1. 模型定义（与训练代码一致）
# =============================================================================
class RetrievalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2. 文件筛选和月度TELSEM读取
# =============================================================================
def parse_filename_info(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        return {
            'date': parts[1][:8],
            'orbit': parts[2][-1],
            'granule_id': os.path.splitext(filename)[0],
            'valid': True
        }
    except (IndexError, ValueError):
        return {'valid': False}


def filter_file(file_info, orbit):
    return (
        file_info['valid']
        and CONFIG['START_DATE'] <= file_info['date'] <= CONFIG['END_DATE']
        and file_info['orbit'] == orbit
    )


def load_monthly_telsem(month):
    pattern = os.path.join(
        CONFIG['TELSEM_DIR'],
        f'AMSR2_MLSE_{month:02d}_5ch*_resampled.nc'
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'TELSEM month {month:02d}: expected 1 file, '
            f'found {len(matches)}: {pattern}'
        )

    with xr.open_dataset(matches[0]) as ds:
        required = [f'E{band}' for band in BANDS_ORDER] + ['lat', 'lon']
        missing = [name for name in required if name not in ds]
        if missing:
            raise ValueError(f'TELSEM variables missing: {missing}')
        print(
            f'Loading TELSEM month {month:02d}: '
            f'{os.path.basename(matches[0])}'
        )
        return ds[[f'E{band}' for band in BANDS_ORDER]].load()


# =============================================================================
# 3. AMSR2亮温读取
# =============================================================================
def get_amsr2_10bands(path):
    with h5py.File(path, 'r') as h5_file:
        lat = h5_file['Latitude of Observation Point for 89A'][:, ::2]
        lon = h5_file['Longitude of Observation Point for 89A'][:, ::2]

        roi = CONFIG['ROI']
        if (
            np.nanmax(lat) < roi['LAT_MIN']
            or np.nanmin(lat) > roi['LAT_MAX']
            or np.nanmax(lon) < roi['LON_MIN']
            or np.nanmin(lon) > roi['LON_MAX']
        ):
            print('   -> Granule out of ROI (Skipping)')
            return None

        channel_names = {
            '10': '10.7GHz',
            '18': '18.7GHz',
            '23': '23.8GHz',
            '36': '36.5GHz',
            '89': '89.0GHz'
        }
        data = {'lat': lat, 'lon': lon}

        for band in BANDS_ORDER:
            dataset_name = (
                f'Brightness Temperature '
                f'(res06,{channel_names[band[:2]]},{band[-1]})'
            )
            if dataset_name not in h5_file:
                raise KeyError(f'Missing AMSR2 channel: {dataset_name}')

            bt = np.asarray(h5_file[dataset_name][:], dtype=np.float32) * 0.01
            if bt.shape != lat.shape:
                raise ValueError(
                    f'Shape mismatch: {dataset_name}: {bt.shape}, lat: {lat.shape}'
                )
            data[f'BT_{band}'] = bt

    return data


# =============================================================================
# 4. Checkpoint读取与检查
# =============================================================================
def load_model(model_name, period_name, use_e):
    suffix = PERIODS[period_name]['MODEL_SUFFIX']
    model_path = os.path.join(CONFIG['MODEL_DIR'], f'{model_name}_{suffix}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    # checkpoint包含sklearn标准化器，仅加载自己训练并信任的文件。
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    expected_feats = [f'BT_{band}' for band in BANDS_ORDER]
    if use_e:
        expected_feats += [f'TELSEM_IN_{band}' for band in BANDS_ORDER]

    feats_in = checkpoint.get('feats_in', checkpoint.get('feats_rad'))
    if feats_in != expected_feats or checkpoint['input_dim'] != len(expected_feats):
        raise ValueError(
            f'{model_name} {period_name}: expected input features '
            f'{expected_feats}, got {feats_in}'
        )
    if checkpoint['target_cols'] != ['T_skin(K)'] or checkpoint['output_dim'] != 1:
        raise ValueError(
            f"{model_name} {period_name}: expected LST-only checkpoint, "
            f"got {checkpoint['target_cols']}"
        )

    scaler_x = checkpoint['scaler_rad']
    scaler_y = checkpoint['scaler_y']
    if scaler_x.n_features_in_ != len(expected_feats):
        raise ValueError(
            f'{model_name} {period_name}: input scaler contains '
            f'{scaler_x.n_features_in_} features, expected {len(expected_feats)}'
        )

    model = RetrievalNet(checkpoint['input_dim'], checkpoint['output_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded: {model_path}')
    return model, scaler_x, scaler_y, feats_in


# =============================================================================
# 5. 单文件推理
# =============================================================================
def run_inference(
    amsr2_file,
    file_info,
    model,
    scaler_x,
    scaler_y,
    feats_in,
    use_e,
    telsem,
    result_dir
):
    print(f'\nProcessing: {os.path.basename(amsr2_file)}')
    amsr2 = get_amsr2_10bands(amsr2_file)
    if amsr2 is None:
        return False

    rows, cols = amsr2['lat'].shape
    lat = amsr2['lat'].ravel()
    lon = amsr2['lon'].ravel()
    roi = CONFIG['ROI']
    roi_mask = (
        (lat >= roi['LAT_MIN'])
        & (lat <= roi['LAT_MAX'])
        & (lon >= roi['LON_MIN'])
        & (lon <= roi['LON_MAX'])
    )

    bt = np.column_stack(
        [amsr2[f'BT_{band}'].ravel() for band in BANDS_ORDER]
    )
    valid_bt = np.isfinite(bt).all(axis=1) & (bt > 0).all(axis=1)
    valid_idx = np.flatnonzero(roi_mask & valid_bt)
    if len(valid_idx) == 0:
        print('   -> No valid BT pixels in ROI.')
        return False

    feature_map = {
        f'BT_{band}': bt[valid_idx, index]
        for index, band in enumerate(BANDS_ORDER)
    }

    if use_e:
        if telsem is None:
            raise ValueError('TELSEM data are required for a BTE model.')

        matched = telsem.sel(
            lat=xr.DataArray(lat[valid_idx], dims='points'),
            lon=xr.DataArray(lon[valid_idx], dims='points'),
            method='nearest'
        )
        e_prior = np.column_stack(
            [matched[f'E{band}'].values for band in BANDS_ORDER]
        )
        valid_e = (
            np.isfinite(e_prior)
            & (e_prior > 0)
            & (e_prior <= 1.0)
        ).all(axis=1)
        valid_idx = valid_idx[valid_e]
        if len(valid_idx) == 0:
            print('   -> No pixels with complete TELSEM priors in ROI.')
            return False

        e_prior = np.clip(e_prior[valid_e], 0.4, 1.0)
        feature_map = {
            f'BT_{band}': bt[valid_idx, index]
            for index, band in enumerate(BANDS_ORDER)
        }
        feature_map.update({
            f'TELSEM_IN_{band}': e_prior[:, index]
            for index, band in enumerate(BANDS_ORDER)
        })

    X = np.column_stack([feature_map[column] for column in feats_in])
    X = scaler_x.transform(X).astype(np.float32)

    preds = []
    batch_size = CONFIG['INFERENCE_BATCH_SIZE']
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            preds.append(model(batch).cpu().numpy())

    lst = scaler_y.inverse_transform(np.concatenate(preds, axis=0))[:, 0]

    result = np.full(rows * cols, np.nan, dtype=np.float32)
    result[valid_idx] = lst
    out_nc = os.path.join(
        result_dir,
        f"Surface_Retrieval_{file_info['granule_id']}.nc"
    )
    ds_out = xr.Dataset(
        {'Retrieved_T_skin_K': (('y', 'x'), result.reshape(rows, cols))},
        coords={
            'lat': (('y', 'x'), amsr2['lat']),
            'lon': (('y', 'x'), amsr2['lon'])
        }
    )
    ds_out.to_netcdf(out_nc)
    print(f'Saved: {out_nc} | Valid LST pixels: {len(valid_idx)}')
    return True


# =============================================================================
# 6. 单模式、单时段推理
# =============================================================================
def run_mode_period(model_name, period_name):
    use_e = MODEL_MODES[model_name]['USE_E']
    orbit = PERIODS[period_name]['ORBIT']
    result_dir = os.path.join(CONFIG['RESULT_ROOT'], model_name, period_name)
    os.makedirs(result_dir, exist_ok=True)

    model, scaler_x, scaler_y, feats_in = load_model(
        model_name, period_name, use_e
    )

    all_files = sorted(glob.glob(os.path.join(CONFIG['AMSR2_DIR'], '*.h5')))
    files = [(path, parse_filename_info(path)) for path in all_files]
    files = [
        (path, info) for path, info in files if filter_file(info, orbit)
    ]
    print(
        f'\nMode: {model_name} | Period: {period_name} | Orbit: {orbit} | '
        f'Matched granules: {len(files)}'
    )
    print(f'Output directory: {result_dir}')

    current_month = None
    telsem = None
    count = 0
    try:
        for amsr2_file, file_info in files:
            if use_e:
                month = int(file_info['date'][4:6])
                if month != current_month:
                    if telsem is not None:
                        telsem.close()
                    telsem = load_monthly_telsem(month)
                    current_month = month

            try:
                count += bool(run_inference(
                    amsr2_file,
                    file_info,
                    model,
                    scaler_x,
                    scaler_y,
                    feats_in,
                    use_e,
                    telsem,
                    result_dir
                ))
            except Exception as exc:
                print(f'Error processing {os.path.basename(amsr2_file)}: {exc}')
    finally:
        if telsem is not None:
            telsem.close()

    print(
        f'Finished: {model_name} {period_name} | '
        f'Successfully processed {count} granules.'
    )
    return count


# =============================================================================
# 7. 主程序：所选模式 × Day/Night
# =============================================================================
if __name__ == '__main__':
    if not SELECTED_MODES:
        raise ValueError('SELECTED_MODES cannot be empty.')

    unknown_modes = [
        mode for mode in SELECTED_MODES if mode not in MODEL_MODES
    ]
    if unknown_modes:
        raise ValueError(
            f'Unknown modes: {unknown_modes}. '
            f'Available modes: {list(MODEL_MODES)}'
        )

    if len(SELECTED_MODES) != len(set(SELECTED_MODES)):
        raise ValueError(f'Duplicate modes in SELECTED_MODES: {SELECTED_MODES}')

    summary = {}
    for selected_mode in SELECTED_MODES:
        for selected_period in PERIODS:
            summary[(selected_mode, selected_period)] = run_mode_period(
                selected_mode, selected_period
            )

    print('\nAll selected inference tasks finished:')
    for (selected_mode, selected_period), count in summary.items():
        print(f'  {selected_mode}/{selected_period}: {count} granules')
