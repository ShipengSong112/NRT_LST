# -*- coding: utf-8 -*-
"""
Four-model LST retrieval comparison with per-epoch noise resampling
(day and night trained separately).

Data are split into 70% training, 20% validation, and 10% clean testing.
Noise-enabled models resample noise for both training and validation each epoch.

Models
------
NN_BT         : 10 BT inputs, no input noise.
NN_BT_Noise   : 10 BT inputs, with per-epoch channel-wise NEΔT noise.
NN_BTE        : 10 BT + 10 emissivity-prior inputs, no input noise.
NN_BTE_Noise  : 10 BT + 10 emissivity-prior inputs, with per-epoch BT and E noise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import copy
import warnings
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 全局配置 (Configuration)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training running on device: {device}")

TASKS = {
    'Day': {
        'DAT_PATH': r'F:/E_LST/CSV/LUT_AMSR2_Fully_Randomized_10channel_A.csv',
        'BT_NOISE_SEED': 20200842,
        'E_NOISE_SEED': 42
    },
    'Night': {
        'DAT_PATH': r'F:/E_LST/CSV/LUT_AMSR2_Fully_Randomized_10channel_D.csv',
        'BT_NOISE_SEED': 20200843,
        'E_NOISE_SEED': 42
    }
}

MODEL_CONFIGS = {
    'NN_BT': {
        'USE_E': False,
        'ADD_BT_NOISE': False,
        'ADD_E_NOISE': False
    },
    'NN_BT_Noise': {
        'USE_E': False,
        'ADD_BT_NOISE': True,
        'ADD_E_NOISE': False
    },
    'NN_BTE': {
        'USE_E': True,
        'ADD_BT_NOISE': False,
        'ADD_E_NOISE': False
    },
    'NN_BTE_Noise': {
        'USE_E': True,
        'ADD_BT_NOISE': True,
        'ADD_E_NOISE': True
    }
}

TRAIN_CONFIG = {
    'BATCH_SIZE': 2048,
    'EPOCHS': 100,
    'LR': 0.001,
    'WEIGHT_DECAY': 1e-4,
    'TRAIN_NOISE_SCALE': 1.0
}

MODEL_DIR = r'F:/E_LST/model'
PLOT_DIR = r'F:/E_LST/JPG/LST/4_model'
BANDS_ORDER = ['10V', '10H', '18V', '18H', '23V', '23H', '36V', '36H', '89V', '89H']

# AMSR2逐通道噪声等效温差（NEΔT，K）
TEMPERATURE_RESOLUTION_K = {
    '10V': 0.70, '10H': 0.70,
    '18V': 0.70, '18H': 0.70,
    '23V': 0.60, '23H': 0.60,
    '36V': 0.70, '36H': 0.70,
    '89V': 1.20, '89H': 1.20,
}
NE_DT_VECTOR_K = np.asarray(
    [TEMPERATURE_RESOLUTION_K[band] for band in BANDS_ORDER],
    dtype=np.float64
)

# TELSEM先验误差标准差（各通道独立高斯扰动）
TELSEM_UNCERTAINTY = {
    '10V': 0.015, '10H': 0.015,
    '18V': 0.015, '18H': 0.015,
    '23V': 0.020, '23H': 0.020,
    '36V': 0.015, '36H': 0.015,
    '89V': 0.030, '89H': 0.030
}


# ---------------------------------------------------------
# 2. 数据与模型定义
# ---------------------------------------------------------
def prepare_data(df, model_config):
    feats_rad = [f'BT_{b}' for b in BANDS_ORDER]
    target_cols = ['T_skin(K)']
    feats_in = list(feats_rad)
    source_cols = []

    if model_config['USE_E']:
        feats_prior = [f'TELSEM_IN_{b}' for b in BANDS_ORDER]
        source_cols = [
            f'TELSEM_{b}' if f'TELSEM_{b}' in df.columns else f'E_{b}'
            for b in BANDS_ORDER
        ]
        feats_in += feats_prior

    required_cols = feats_rad + source_cols + target_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required_cols, inplace=True)

    if model_config['USE_E']:
        for band, base_col in zip(BANDS_ORDER, source_cols):
            e_values = df[base_col].to_numpy(dtype=np.float64, copy=True)
            df[f'TELSEM_IN_{band}'] = np.clip(e_values, 0.4, 1.0)

    print(
        f"Input features: {len(feats_in)}; Target: LST only; "
        f"Valid samples: {len(df)}"
    )
    return df, feats_rad, feats_in, target_cols, source_cols


class CloudAwareDataset(Dataset):
    def __init__(self, X_rad, Y, mode='train'):
        self.X_rad = X_rad
        self.Y = Y
        self.mode = mode

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_rad[idx].clone(), self.Y[idx]


class RetrievalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RetrievalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 3. 核心训练函数
# ---------------------------------------------------------
def train_and_evaluate(model_name, model_config, task_name, paths_dict):
    print(f"\n{'='*50}\nStarting Task: {model_name} - {task_name}\n{'='*50}")

    dat_path = paths_dict['DAT_PATH']
    save_path = os.path.join(
        MODEL_DIR, f'{model_name}_{task_name.lower()}_dynamic_noise.pth'
    )

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"File not found at {dat_path}")

    df = pd.read_csv(dat_path)
    df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]

    df, feats_rad, feats_in, target_cols, source_cols = prepare_data(
        df, model_config
    )

    idx_train, idx_remaining = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_remaining, test_size=1/3, random_state=42
    )
    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    print(
        f"Split samples - Train: {len(train_df)}, Validation: {len(val_df)}, "
        f"Test: {len(test_df)}"
    )

    # scaler_rad名称沿用旧checkpoint；BTE模型中实际包含全部20个输入特征。
    scaler_rad = StandardScaler().fit(train_df[feats_in].values)
    scaler_y = StandardScaler().fit(train_df[target_cols].values)

    X_train_raw_clean = train_df[feats_in].values.astype(np.float64, copy=True)
    X_rad_train = scaler_rad.transform(X_train_raw_clean)
    Y_train = scaler_y.transform(train_df[target_cols].values)
    X_rad_val = scaler_rad.transform(val_df[feats_in].values)
    Y_val = scaler_y.transform(val_df[target_cols].values)
    X_rad_test = scaler_rad.transform(test_df[feats_in].values)
    Y_test = scaler_y.transform(test_df[target_cols].values)

    tensors_train = [
        torch.tensor(x, dtype=torch.float32) for x in [X_rad_train, Y_train]
    ]
    tensors_val = [
        torch.tensor(x, dtype=torch.float32) for x in [X_rad_val, Y_val]
    ]
    tensors_test = [
        torch.tensor(x, dtype=torch.float32) for x in [X_rad_test, Y_test]
    ]

    train_loader = DataLoader(
        CloudAwareDataset(*tensors_train, 'train'),
        batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        CloudAwareDataset(*tensors_val, 'val'),
        batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        CloudAwareDataset(*tensors_test, 'test'),
        batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=False, pin_memory=True
    )

    input_dim = len(feats_in)
    output_dim = len(target_cols)
    model = RetrievalNet(input_dim, output_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=TRAIN_CONFIG['LR'],
        weight_decay=TRAIN_CONFIG['WEIGHT_DECAY']
    )
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    noise_scale = TRAIN_CONFIG['TRAIN_NOISE_SCALE']
    bt_indices_list = [feats_in.index(col) for col in feats_rad]
    bt_indices = torch.tensor(bt_indices_list, dtype=torch.long, device=device)
    bt_noise_std = torch.tensor(
        NE_DT_VECTOR_K * noise_scale / scaler_rad.scale_[bt_indices_list],
        dtype=torch.float32, device=device
    )
    bt_train_rng = torch.Generator(device=device)
    bt_train_rng.manual_seed(paths_dict['BT_NOISE_SEED'])
    bt_val_rng = torch.Generator(device=device)
    bt_val_rng.manual_seed(paths_dict['BT_NOISE_SEED'] + 100000)

    e_indices_list = [
        feats_in.index(f'TELSEM_IN_{band}')
        for band in BANDS_ORDER if model_config['USE_E']
    ]
    if e_indices_list:
        e_indices = torch.tensor(e_indices_list, dtype=torch.long, device=device)
        e_noise_std = torch.tensor(
            np.asarray(
                [TELSEM_UNCERTAINTY[band] for band in BANDS_ORDER],
                dtype=np.float64
            )
            / scaler_rad.scale_[e_indices_list],
            dtype=torch.float32, device=device
        )
        e_clip_min = torch.tensor(
            (0.4 - scaler_rad.mean_[e_indices_list])
            / scaler_rad.scale_[e_indices_list],
            dtype=torch.float32, device=device
        )
        e_clip_max = torch.tensor(
            (1.0 - scaler_rad.mean_[e_indices_list])
            / scaler_rad.scale_[e_indices_list],
            dtype=torch.float32, device=device
        )
        e_train_rng = torch.Generator(device=device)
        e_train_rng.manual_seed(paths_dict['E_NOISE_SEED'])
        e_val_rng = torch.Generator(device=device)
        e_val_rng.manual_seed(paths_dict['E_NOISE_SEED'] + 100000)

    def add_dynamic_noise(batch_x, bt_rng, e_rng=None):
        if model_config['ADD_BT_NOISE']:
            bt_noise = torch.randn(
                (len(batch_x), len(bt_indices_list)),
                generator=bt_rng, device=device
            ) * bt_noise_std
            batch_x[:, bt_indices] = batch_x[:, bt_indices] + bt_noise

        if model_config['ADD_E_NOISE']:
            e_values = batch_x[:, e_indices] + torch.randn(
                (len(batch_x), len(e_indices_list)),
                generator=e_rng, device=device
            ) * e_noise_std
            batch_x[:, e_indices] = torch.maximum(
                torch.minimum(e_values, e_clip_max), e_clip_min
            )

        return batch_x

    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx = add_dynamic_noise(
                bx, bt_train_rng,
                e_train_rng if model_config['ADD_E_NOISE'] else None
            )

            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                bx = add_dynamic_noise(
                    bx, bt_val_rng,
                    e_val_rng if model_config['ADD_E_NOISE'] else None
                )
                pred = model(bx)
                loss = loss_fn(pred, by)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        if avg_val < best_loss:
            best_loss = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1:03d}: Train Loss={np.mean(train_losses):.5f} | "
                f"Val Loss={avg_val:.5f}"
            )

    print(f"[{model_name} - {task_name}] Best Val Loss: {best_loss:.5f}")

    model.load_state_dict(best_model_wts)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_rad': scaler_rad,
        'scaler_y': scaler_y,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'feats_rad': feats_rad,
        'feats_in': feats_in,
        'target_cols': target_cols,
        'model_name': model_name,
        'model_config': model_config,
        'source_emissivity_cols': source_cols,
        'config': TRAIN_CONFIG,
        'temperature_resolution_K': TEMPERATURE_RESOLUTION_K,
        'tel_sem_uncertainty': TELSEM_UNCERTAINTY,
        'bt_noise_seed': paths_dict['BT_NOISE_SEED'],
        'e_noise_seed': paths_dict['E_NOISE_SEED'],
        'validation_noise_seed_offset': 100000,
        'data_split': {'train': 0.7, 'validation': 0.2, 'test': 0.1},
        'noise_sampling': 'per_epoch_training_and_validation',
        'test_noise': 'none'
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pred = model(bx)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(by.numpy())

    preds_real = scaler_y.inverse_transform(np.concatenate(preds_list, axis=0))
    trues_real = scaler_y.inverse_transform(np.concatenate(trues_list, axis=0))

    del df, train_df, val_df, test_df
    del tensors_train, tensors_val, tensors_test
    gc.collect()

    return trues_real[:, 0], preds_real[:, 0]


# ---------------------------------------------------------
# 4. SCI风格联合绘图函数
# ---------------------------------------------------------
def plot_sci_four_model_results(results_dict, save_path, task_name):
    print(f"\nGenerating four-model test plot for {task_name}...")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
    plt.subplots_adjust(wspace=0.15, hspace=0.20)

    model_order = ['NN_BT', 'NN_BT_Noise', 'NN_BTE', 'NN_BTE_Noise']
    labels = ['(a)', '(b)', '(c)', '(d)']
    hb_list = []
    global_vmin, global_vmax = 220, 340

    for i, model_name in enumerate(model_order):
        ax = axes.flat[i]
        y_true, y_pred = results_dict[model_name]

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))
        bias = np.mean(y_pred - y_true)
        r_val = np.corrcoef(y_true, y_pred)[0, 1]
        slope, intercept = np.polyfit(y_true, y_pred, 1)

        hb = ax.hexbin(
            y_true, y_pred, gridsize=60, cmap='Blues', mincnt=1, bins='log',
            extent=(global_vmin, global_vmax, global_vmin, global_vmax)
        )
        hb_list.append(hb)

        ax.plot(
            [global_vmin, global_vmax], [global_vmin, global_vmax],
            'k--', lw=1.5, label='1:1 Line'
        )
        fit_line_x = np.array([global_vmin, global_vmax])
        fit_line_y = slope * fit_line_x + intercept
        ax.plot(fit_line_x, fit_line_y, 'r-', lw=1.5, label='Linear Fit')

        ax.set_xlim(global_vmin, global_vmax)
        ax.set_ylim(global_vmin, global_vmax)
        ax.set_box_aspect(1)

        sign = '+' if intercept >= 0 else '-'
        stats_text = (
            f"Fit: $y = {slope:.2f}x {sign} {abs(intercept):.2f}$\n"
            f"RMSE: {rmse:.3f} K\n"
            f"MAE: {mae:.3f} K\n"
            f"Bias: {bias:.3f} K\n"
            f"$R$: {r_val:.3f}"
        )
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='square,pad=0.5', fc='white', ec='black', alpha=0.9)
        )

        ax.legend(loc='lower right', frameon=False, fontsize=11)
        ax.set_title(
            f"{labels[i]} {model_name}", fontsize=14,
            fontweight='bold', loc='left'
        )
        if i >= 2:
            ax.set_xlabel('True LST (K)', fontsize=12, fontweight='bold')
        if i % 2 == 0:
            ax.set_ylabel('Retrieved LST (K)', fontsize=12, fontweight='bold')
        ax.tick_params(
            axis='both', which='major', labelsize=11,
            direction='in', top=True, right=True
        )

    max_log_count = max(hb.get_array().max() for hb in hb_list)
    min_log_count = min(hb.get_array().min() for hb in hb_list)
    if min_log_count <= 0:
        min_log_count = 1
    for hb in hb_list:
        hb.set_clim(vmin=min_log_count, vmax=max_log_count)

    cb = fig.colorbar(
        hb_list[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.03
    )
    cb.set_label('$\\log_{10}(N)$', fontsize=12)
    cb.ax.tick_params(labelsize=11, direction='in')
    fig.suptitle(f'{task_name}time LST - Clean Test Set', fontsize=15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"\nFinal four-model test plot saved to: {save_path}")


# ---------------------------------------------------------
# 5. 主执行逻辑：4种配置 × 白天/黑夜 = 8个模型
# ---------------------------------------------------------
if __name__ == '__main__':
    all_results = {task_name: {} for task_name in TASKS}

    for model_name, model_config in MODEL_CONFIGS.items():
        for task_name, paths in TASKS.items():
            y_true, y_pred = train_and_evaluate(
                model_name, model_config, task_name, paths
            )
            all_results[task_name][model_name] = (y_true, y_pred)

    for task_name in TASKS:
        save_plot_path = os.path.join(
            PLOT_DIR,
            f'test_4_model_compare_{task_name.lower()}_dynamic_noise.jpg'
        )
        plot_sci_four_model_results(
            all_results[task_name], save_plot_path, task_name
        )
