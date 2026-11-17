# -*- coding: utf-8 -*-
"""
Full-Stack Physics-Informed Retrieval Training (LST Only Edition - Day & Night Combined)
------------------------------------------------------------------
更新说明:
1. [VISUALIZATION] 提取全局唯一图例放置在图像顶部，将各自的拟合方程移入信息面板中，使排版符合SCI标准。
2. [OUTPUT] 统一输出至 shap 文件夹，新增评估指标 CSV。
3. [ANALYSIS] 绘制 2x2 SHAP 分析图，按波段重要性自动排序，散点图隐去纵轴标注与柱状图对齐。
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import shap
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
        'SAVE_PATH': r'F:/E_LST/model/lst_only_retrieval_checkpoint_final_10channel_day.pth'
    },
    'Night': {
        'DAT_PATH': r'F:/E_LST/CSV/LUT_AMSR2_Fully_Randomized_10channel_D.csv',
        'SAVE_PATH': r'F:/E_LST/model/lst_only_retrieval_checkpoint_final_10channel_night.pth'
    }
}

TRAIN_CONFIG = {
    'BATCH_SIZE': 2048,
    'EPOCHS': 100,
    'LR': 0.001,
    'WEIGHT_DECAY': 1e-4
}

# False reloads the current checkpoint and regenerates the diagnostics without
# changing the trained model. Set True only when the model should be retrained.
RETRAIN_MODEL = False

BANDS_ORDER = ['10V', '10H', '18V', '18H', '23V', '23H', '36V', '36H', '89V', '89H']

# --- 输出路径统一配置 ---
OUT_DIR = r'F:\E_LST\JPG\LST\5_shap'
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PLOT_PATH = os.path.join(OUT_DIR, 'train_lst_only_combined_day_night.jpg')
SAVE_METRICS_CSV = os.path.join(OUT_DIR, 'evaluation_metrics.csv')
SAVE_SHAP_CSV = os.path.join(OUT_DIR, 'shap_importance_metrics.csv')
SAVE_SHAP_PLOT = os.path.join(OUT_DIR, 'shap_four_panel_analysis.jpg')
SHAP_CACHE_PATH = os.path.join(OUT_DIR, 'shap_plot_cache.npz')
USE_SHAP_CACHE = os.environ.get('BAND_SENSITIVE_USE_CACHE') == '1'

# ---------------------------------------------------------
# 2. 数据与模型定义
# ---------------------------------------------------------
def prepare_data(df):
    feats_rad = [f'BT_{b}' for b in BANDS_ORDER]
    target_cols = ['T_skin(K)']
    all_required = list(set(feats_rad + target_cols))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=all_required, inplace=True)
    return df, feats_rad, target_cols

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
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, output_dim) 
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 3. 核心训练与SHAP计算函数
# ---------------------------------------------------------
def train_and_evaluate(task_name, paths_dict):
    print(f"\n{'='*50}\nStarting Task: {task_name}\n{'='*50}")
    
    dat_path = paths_dict['DAT_PATH']
    save_path = paths_dict['SAVE_PATH']
    
    if not os.path.exists(dat_path): 
        raise FileNotFoundError(f"File not found at {dat_path}")
        
    df = pd.read_csv(dat_path)
    df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    
    df, feats_rad, target_cols = prepare_data(df)
    
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_df = df.iloc[idx_train].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    
    checkpoint = None
    if RETRAIN_MODEL:
        scaler_rad = StandardScaler().fit(train_df[feats_rad].values)
        scaler_y = StandardScaler().fit(train_df[target_cols].values)
        input_dim = len(feats_rad)
        output_dim = len(target_cols)
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found at {save_path}")
        try:
            checkpoint = torch.load(
                save_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(save_path, map_location=device)
        if checkpoint.get('feats_rad') != feats_rad:
            raise ValueError(
                f"Checkpoint feature order does not match {task_name} inputs."
            )
        scaler_rad = checkpoint['scaler_rad']
        scaler_y = checkpoint['scaler_y']
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
    
    X_rad_train = scaler_rad.transform(train_df[feats_rad].values)
    Y_train = scaler_y.transform(train_df[target_cols].values)
    X_rad_test = scaler_rad.transform(test_df[feats_rad].values)
    Y_test = scaler_y.transform(test_df[target_cols].values)
    
    tensors_test = [torch.tensor(x, dtype=torch.float32) for x in [X_rad_test, Y_test]]
    test_loader = DataLoader(CloudAwareDataset(*tensors_test, 'test'), 
                             batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=False, pin_memory=True)

    model = RetrievalNet(input_dim, output_dim).to(device)

    if RETRAIN_MODEL:
        tensors_train = [
            torch.tensor(x, dtype=torch.float32)
            for x in [X_rad_train, Y_train]
        ]
        train_loader = DataLoader(
            CloudAwareDataset(*tensors_train, 'train'),
            batch_size=TRAIN_CONFIG['BATCH_SIZE'],
            shuffle=True,
            pin_memory=True,
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['LR'],
            weight_decay=TRAIN_CONFIG['WEIGHT_DECAY'],
        )
        loss_fn = nn.MSELoss()
        best_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(TRAIN_CONFIG['EPOCHS']):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    loss = loss_fn(pred, by)
                    val_losses.append(loss.item())

            avg_val = np.mean(val_losses)
            if avg_val < best_loss:
                best_loss = avg_val
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f"[{task_name}] Best Val Loss: {best_loss:.5f}")
        model.load_state_dict(best_model_wts)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'scaler_rad': scaler_rad,
            'scaler_y': scaler_y,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'feats_rad': feats_rad,
            'target_cols': target_cols,
            'config': TRAIN_CONFIG
        }
        torch.save(checkpoint, save_path)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[{task_name}] Loaded checkpoint without retraining.")
    
    model.eval()
    if RETRAIN_MODEL:
        preds_list, trues_list = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                pred = model(bx)
                preds_list.append(pred.cpu().numpy())
                trues_list.append(by.numpy())

        preds_real = scaler_y.inverse_transform(
            np.concatenate(preds_list, axis=0)
        )
        trues_real = scaler_y.inverse_transform(
            np.concatenate(trues_list, axis=0)
        )
    else:
        preds_real = None
        trues_real = None
    
    print(f"[{task_name}] Calculating SHAP values...")
    bg_samples = torch.tensor(X_rad_train[:500], dtype=torch.float32).to(device)
    test_samples_shap = torch.tensor(X_rad_test[:1000], dtype=torch.float32).to(device)
    test_features_shap = X_rad_test[:1000]

    explainer = shap.DeepExplainer(model, bg_samples)
    shap_values = explainer.shap_values(test_samples_shap)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")
        
    del df, train_df, test_df, tensors_test
    gc.collect()
    
    if RETRAIN_MODEL:
        return (
            trues_real[:, 0],
            preds_real[:, 0],
            shap_values,
            test_features_shap,
        )
    return None, None, shap_values, test_features_shap

# ---------------------------------------------------------
# 4. 可视化与评估指标提取函数
# ---------------------------------------------------------
def plot_sci_combined_results(results_dict, save_plot_path, save_csv_path):
    print("\nGenerating SCI-style Model Evaluation Plot and Metrics CSV...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.15) 
    
    tasks = ['Day', 'Night']
    labels = ['(a)', '(b)']
    hb_list = [] 
    metrics_list = [] 
    
    global_vmin, global_vmax = 220, 340

    for i, task in enumerate(tasks):
        ax = axes[i]
        y_true, y_pred, _, _ = results_dict[task]
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))
        bias = np.mean(y_pred - y_true)
        r_val = np.corrcoef(y_true, y_pred)[0, 1]
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        
        metrics_list.append({
            'Task': task,
            'RMSE (K)': round(rmse, 4),
            'MAE (K)': round(mae, 4),
            'Bias (K)': round(bias, 4),
            'R': round(r_val, 4),
            'Slope': round(slope, 4),
            'Intercept': round(intercept, 4)
        })

        cmap = 'Blues'
        hb = ax.hexbin(y_true, y_pred, gridsize=60, cmap=cmap, mincnt=1, bins='log',
                       extent=(global_vmin, global_vmax, global_vmin, global_vmax))
        hb_list.append(hb)
        
        ax.plot([global_vmin, global_vmax], [global_vmin, global_vmax], 'k--', lw=1.5, label='1:1 Line')
        
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
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', 
                fontsize=11, bbox=dict(boxstyle='square,pad=0.5', fc='white', ec='black', alpha=0.9))
        
        ax.legend(loc='lower right', frameon=False, fontsize=11)
        
        ax.set_title(f"{labels[i]} {task} LST", fontsize=14, fontweight='bold', loc='left')
        ax.set_xlabel('True LST (K)', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Retrieved LST (K)', fontsize=12, fontweight='bold')
            
        ax.tick_params(axis='both', which='major', labelsize=11, direction='in', top=True, right=True)

    max_log_count = max(hb_list[0].get_array().max(), hb_list[1].get_array().max())
    min_log_count = min(hb_list[0].get_array().min(), hb_list[1].get_array().min())
    
    if min_log_count <= 0: min_log_count = 1
        
    for hb in hb_list: hb.set_clim(vmin=min_log_count, vmax=max_log_count)

    cb = fig.colorbar(hb_list[1], ax=axes, fraction=0.02, pad=0.03)
    cb.set_label(r'$\log_{10}(N)$', fontsize=12)
    cb.ax.tick_params(labelsize=11, direction='in')

    plt.savefig(save_plot_path, bbox_inches='tight', pad_inches=0.1)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(save_csv_path, index=False, encoding='utf-8-sig')

def calculate_and_save_shap_metrics(results_dict, save_csv_path):
    """计算特征的平均重要性及百分比，并输出为 CSV 表格"""
    print(f"\nCalculating SHAP importance metrics and saving to {save_csv_path}...")
    metrics = []
    
    for task_name in ['Day', 'Night']:
        shap_values = results_dict[task_name][2]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        importance_pct = (mean_abs_shap / total_shap) * 100
        
        for i, band in enumerate(BANDS_ORDER):
            metrics.append({
                'Task': task_name,
                'Band': band,
                'Mean_Abs_SHAP': round(mean_abs_shap[i], 5),
                'Importance_Percent(%)': round(importance_pct[i], 3)
            })
            
    df_shap = pd.DataFrame(metrics)
    df_shap.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    return df_shap

def plot_four_panel_shap_analysis(results_dict, save_path):
    """绘制 SCI 规范的 2x2 SHAP 分析图 (散点图隐去 Y 轴标签并共享全局 Colorbar)"""
    print("\nGenerating Unified 2x2 SCI-style SHAP Subplots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    tasks = ['Day', 'Night']
    plot_letters = [['(a)', '(b)'], ['(c)', '(d)']]
    
    for i, task_name in enumerate(tasks):
        shap_values = results_dict[task_name][2]
        test_features = results_dict[task_name][3]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_order = np.argsort(mean_abs_shap)[::-1]
        y_positions = np.arange(len(feature_order))
        
        # --- 左侧子图: 全局重要性 (Bar Plot) ---
        ax_bar = axes[i, 0]
        ax_bar.barh(
            y_positions,
            mean_abs_shap[feature_order],
            color='#4575b4',
        )
        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(
            [BANDS_ORDER[index] for index in feature_order],
            fontsize=11,
        )
        ax_bar.invert_yaxis()
        ax_bar.set_title(f"{plot_letters[i][0]} {task_name} Feature Global Importance", fontsize=13, fontweight='bold', loc='left')
        ax_bar.set_xlabel('Mean absolute SHAP value', fontsize=11)
        ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
        
        # --- 右侧子图: 散点分布 (Dot Plot) ---
        ax_dot = axes[i, 1]
        rng = np.random.default_rng(20200827 + i)
        for rank, feature_index in enumerate(feature_order):
            feature_values = test_features[:, feature_index]
            feature_min = np.nanmin(feature_values)
            feature_max = np.nanmax(feature_values)
            if feature_max > feature_min:
                normalized_feature = (
                    (feature_values - feature_min)
                    / (feature_max - feature_min)
                )
            else:
                normalized_feature = np.full(feature_values.shape, 0.5)
            jitter = rng.normal(0.0, 0.075, size=shap_values.shape[0])
            ax_dot.scatter(
                shap_values[:, feature_index],
                rank + jitter,
                c=normalized_feature,
                cmap='coolwarm',
                vmin=0.0,
                vmax=1.0,
                s=13,
                alpha=0.75,
                linewidths=0.0,
                rasterized=True,
            )
            ax_dot.axhline(
                rank,
                color='0.85',
                linestyle=':',
                linewidth=0.7,
                zorder=0,
            )
        ax_dot.axvline(0.0, color='0.55', linewidth=1.0)
        ax_dot.set_ylim(-0.5, len(feature_order) - 0.5)
        ax_dot.invert_yaxis()
        ax_dot.set_title(f"{plot_letters[i][1]} {task_name} Feature SHAP Scatter", fontsize=13, fontweight='bold', loc='left')
        ax_dot.set_xlabel('SHAP Value', fontsize=11)
        
        # 清除右侧子图的 Y 轴标签和刻度，以确保纵向对齐感
        ax_dot.set_ylabel('')
        ax_dot.tick_params(axis='y', left=False, labelleft=False)

    # --- 添加全局共享的连续刻度 Colorbar ---
    sm = cm.ScalarMappable(cmap='coolwarm', norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    # 将 Colorbar 绑定至散点图所在的列列
    cbar = fig.colorbar(sm, ax=axes[:, 1].tolist(), fraction=0.03, pad=0.04)
    cbar.set_label('Normalized Feature Value', fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"2x2 SHAP analysis plot saved to: {save_path}")

# ---------------------------------------------------------
# 5. 主执行逻辑
# ---------------------------------------------------------
if __name__ == '__main__':
    if USE_SHAP_CACHE:
        if not os.path.exists(SHAP_CACHE_PATH):
            raise FileNotFoundError(
                f"SHAP plotting cache not found at {SHAP_CACHE_PATH}"
            )
        with np.load(SHAP_CACHE_PATH) as cache:
            all_results = {
                task_name: (
                    None,
                    None,
                    cache[f'{task_name}_shap'],
                    cache[f'{task_name}_features'],
                )
                for task_name in TASKS
            }
        print(f"Loaded SHAP plotting cache: {SHAP_CACHE_PATH}")
    else:
        all_results = {}
        for task_name, paths in TASKS.items():
            y_true, y_pred, shap_vals, test_feats = train_and_evaluate(
                task_name,
                paths,
            )
            all_results[task_name] = (
                y_true,
                y_pred,
                shap_vals,
                test_feats,
            )
        np.savez_compressed(
            SHAP_CACHE_PATH,
            **{
                f'{task_name}_{array_name}': all_results[task_name][index]
                for task_name in TASKS
                for array_name, index in (('shap', 2), ('features', 3))
            },
        )
        print(f"Saved SHAP plotting cache: {SHAP_CACHE_PATH}")
        
    # 1. 仅在重新训练时更新反演拟合图和基本评估 CSV。
    if RETRAIN_MODEL and not USE_SHAP_CACHE:
        plot_sci_combined_results(
            all_results,
            SAVE_PLOT_PATH,
            SAVE_METRICS_CSV,
        )
    else:
        print(
            "Checkpoint plotting mode: keeping the existing model "
            "evaluation plot and CSV."
        )
    
    # 2. 计算特征重要性指标并输出至 CSV
    df_shap_metrics = calculate_and_save_shap_metrics(all_results, SAVE_SHAP_CSV)
    
    # 3. 生成 2x2 SHAP 分析图
    plot_four_panel_shap_analysis(all_results, SAVE_SHAP_PLOT)

