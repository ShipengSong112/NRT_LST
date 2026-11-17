# -*- coding: utf-8 -*-
"""
Part 2: LST Inference and In-situ Validation (LST Only Edition)
功能: 
1. 通过仅依赖亮温输入的 RetrievalNet 模型进行 LST 推理。
2. 提取 A 轨（Day）和/或 D 轨（Night）数据进行匹配验证。
3. 动态绘制散点图以及时间序列图，站点标注使用拼音。
4. 评价指标包含 N, RMSE, MAE, Bias, R2。
5. 自动导出散点图与时间序列图。
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 字体与绘图基础设置 (SCI 风格)
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# =============================================================================
# 2. 配置区域
# =============================================================================
CONFIG = {
    'INPUT_BT_CSV': r'F:\E_LST\CSV\Extracted_AMSR2_BT_ALL.csv',
    'STATION_INFO_CSV': r'G:\LST\CSV\station\heihe.csv',
    'STATION_DATA_CSV': r'G:\LST\CSV\station\heihe_LST.csv',
    
    # 轨道选择列表: 可改为 ['A'], ['D'], 或 ['A', 'D']
    # 注意：如果仅训练了白天(Day)模型，请修改为 ['A']
    'SELECTED_ORBITS': ['A', 'D'], 
    
    'MODEL_PATHS': {
        # 适配代码一中的单变量 (LST Only) 检查点命名
        'A': r'F:\E_LST\model\NN_BT_day_dynamic_noise.pth',
        'D': r'F:\E_LST\model\NN_BT_night_dynamic_noise.pth'
    },
    
    # 图片导出路径
    'SAVE_DIR': r'F:\E_LST\JPG\LST\8_site',
    
    'TIME_WINDOW_MIN': 15,
    'TARGET_NAMES_CN': ['大满', '花寨子', '混合林'],
    'PINYIN_MAP': {'大满': 'Daman', '花寨子': 'Huazhaizi', '混合林': 'Hunhelin'}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 3. 辅助函数 (指标计算与绘图单位)
# =============================================================================
def calc_metrics(x, y):
    """
    计算评估指标。
    x: 观测值 (In-situ LST)
    y: 预测值 (AMSR2 LST)
    Bias 计算方式为: 预测值均值 - 观测值均值 (Mean(y - x))
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 2:
        return None, None, None, None, 0
    
    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)
    bias = np.mean(y - x)
    r2 = r2_score(x, y)
    
    return rmse, mae, bias, r2, len(x)

def plot_scatter_unit(ax, x, y, title, xlabel, ylabel):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = np.asarray(x)[mask]
    y_valid = np.asarray(y)[mask]

    if len(x_valid) < 2:
        ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=14, pad=8)
        return

    rmse, mae, bias, r2, n = calc_metrics(x_valid, y_valid)

    ax.scatter(x_valid, y_valid, s=30, c='#1f77b4', alpha=0.7, edgecolors='k', linewidths=0.5)

    vmin = min(x_valid.min(), y_valid.min())
    vmax = max(x_valid.max(), y_valid.max())
    buff = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
    
    # 1:1 线
    ax.plot([vmin - buff, vmax + buff], [vmin - buff, vmax + buff], 'k--', lw=1.5)
    
    # 拟合线
    z = np.polyfit(x_valid, y_valid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(vmin, vmax, 100)
    ax.plot(x_line, p(x_line), 'r-', lw=1.5)

    stats_text = (f'N = {n}\n'
                  f'RMSE = {rmse:.2f} K\n'
                  f'MAE = {mae:.2f} K\n'
                  f'Bias = {bias:.2f} K\n'
                  f'$R^2$ = {r2:.2f}')

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim([vmin - buff, vmax + buff])
    ax.set_ylim([vmin - buff, vmax + buff])
    ax.set_box_aspect(1)

# =============================================================================
# 4. 模型定义与推理引擎 (适配代码一)
# =============================================================================
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

class InferenceEngine:
    def __init__(self, model_path):
        self.checkpoint = torch.load(model_path, map_location=device)
        self.model = self._load_model()
        self.scaler_rad = self.checkpoint['scaler_rad']
        self.scaler_y = self.checkpoint['scaler_y']
        # 严格使用训练保存的波段顺序提取特征
        self.bands = [b.split('_')[-1] for b in self.checkpoint.get('feats_rad', [])] 
        if not self.bands:
            self.bands = ['10V', '10H', '18V', '18H', '23V', '23H', '36V', '36H', '89V', '89H']
        self.target_cols = self.checkpoint.get('target_cols', ['T_skin(K)'])

    def _load_model(self):
        # 使用与代码一相同的 RetrievalNet
        model = RetrievalNet(self.checkpoint['input_dim'], self.checkpoint['output_dim']).to(device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, df_orbit):
        X_rad = df_orbit[[f'BT_{b}' for b in self.bands]].values
        X_in = self.scaler_rad.transform(X_rad)
        X_tensor = torch.tensor(X_in, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_norm = self.model(X_tensor).cpu().numpy()
        preds_real = self.scaler_y.inverse_transform(preds_norm)
        lst_idx = self.target_cols.index('T_skin(K)') if 'T_skin(K)' in self.target_cols else 0
        return preds_real[:, lst_idx]

# =============================================================================
# 5. 主逻辑
# =============================================================================
def main():
    # --- A. 加载站点元数据，用于后续绘图映射 ---
    meta_df = pd.read_csv(CONFIG['STATION_INFO_CSV'], encoding='gb18030')
    target_meta = meta_df[meta_df['name'].isin(CONFIG['TARGET_NAMES_CN'])].copy()
    target_meta['name'] = pd.Categorical(target_meta['name'], categories=CONFIG['TARGET_NAMES_CN'], ordered=True)
    target_meta = target_meta.sort_values('name')

    name_map_en2cn = dict(zip(target_meta['station'], target_meta['name']))
    ordered_stations_en = target_meta['station'].tolist()

    # --- B. 加载并推理 AMSR2 轨道数据 ---
    print("\n[1/4] 提取并推理 AMSR2 轨道数据...")
    df_sat = pd.read_csv(CONFIG['INPUT_BT_CSV'], encoding='gb18030')
    df_sat['datetime'] = pd.to_datetime(df_sat['datetime'])
    df_sat = df_sat[df_sat['orbit'].isin(CONFIG['SELECTED_ORBITS'])].copy()
    
    results = []
    for orbit in CONFIG['SELECTED_ORBITS']:
        print(f"  正在使用模型推理轨道: {orbit}...")
        try:
            engine = InferenceEngine(CONFIG['MODEL_PATHS'][orbit])
            df_orbit = df_sat[df_sat['orbit'] == orbit].copy()
            df_orbit['amsr2_val'] = engine.predict(df_orbit)
            results.append(df_orbit)
        except Exception as e:
            print(f"  [Error] 轨道 {orbit} 模型加载失败: {e}。请检查路径或仅运行可用轨道。")
    
    if not results:
        print("无成功推理的数据。程序退出。")
        return
        
    df_final_sat = pd.concat(results)

    # --- C. 时间匹配 (In-situ) ---
    print("\n[2/4] 读取并匹配 In-situ 观测数据...")
    obs_df = pd.read_csv(CONFIG['STATION_DATA_CSV'], encoding='gb18030')
    obs_df = obs_df[obs_df['station'].isin(ordered_stations_en)].copy()
    obs_df['datetime'] = pd.to_datetime(obs_df[['year', 'month', 'day', 'hour', 'minute']])
    
    df_final_sat['station_val'] = np.nan
    obs_grouped = obs_df.groupby('station')

    for idx, row in df_final_sat.iterrows():
        st, t0 = row['station'], row['datetime']
        if st in obs_grouped.groups:
            obs_sub = obs_grouped.get_group(st)
            mask = (obs_sub['datetime'] >= t0 - timedelta(minutes=CONFIG['TIME_WINDOW_MIN'])) & \
                   (obs_sub['datetime'] <= t0 + timedelta(minutes=CONFIG['TIME_WINDOW_MIN']))
            if len(obs_sub[mask]) >= 2:
                df_final_sat.at[idx, 'station_val'] = obs_sub[mask]['LST'].mean()

    df_final = df_final_sat.dropna(subset=['station_val', 'amsr2_val']).copy()
    print(f"  匹配成功，总样本量: {len(df_final)}")

    if df_final.empty:
        print("[Error] 匹配后无有效数据，程序退出。")
        return
        
    # 创建导出目录
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)

    # ==========================================================
    # --- D. 绘图 1：散点图 (动态自适应布局) ---
    # ==========================================================
    print("\n[3/4] 绘制验证散点图...")
    orbits_plot = df_final['orbit'].unique().tolist()
    nrows = len(orbits_plot)
    ncols = len(ordered_stations_en)
    
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=300)
    # 强制将 axes 转换为 2D，防止 nrows=1 或 ncols=1 时引发索引错误
    axes1 = np.atleast_2d(axes1).reshape(nrows, ncols) 
    
    for row_idx, orbit in enumerate(orbits_plot):
        orbit_label = "Day (A)" if orbit == 'A' else "Night (D)"
        for col_idx, st_en in enumerate(ordered_stations_en):
            ax = axes1[row_idx, col_idx]
            st_cn = name_map_en2cn[st_en]
            st_pinyin = CONFIG['PINYIN_MAP'].get(st_cn, st_en)
            
            df_plot = df_final[(df_final['orbit'] == orbit) & (df_final['station'] == st_en)]
            
            title = f"({chr(97 + row_idx * ncols + col_idx)}) {st_pinyin} - {orbit_label}"
            plot_scatter_unit(
                ax, 
                df_plot['station_val'], df_plot['amsr2_val'],
                title=title, 
                xlabel="In-situ LST (K)", 
                ylabel="AMSR2 LST (K)"
            )

    plt.tight_layout()
    scatter_save_path = os.path.join(CONFIG['SAVE_DIR'], 'LST_Validation_Scatter.jpg')
    fig1.savefig(scatter_save_path, dpi=300, bbox_inches='tight')
    print(f"  散点图已导出至: {scatter_save_path}")
    plt.show()

    # ==========================================================
    # --- E. 绘图 2：时间序列图 ---
    # ==========================================================
    print("\n[4/4] 绘制时间序列图...")
    df_ts = df_final.sort_values(by=['station', 'datetime'])

    fig2, axes2 = plt.subplots(ncols, 1, figsize=(16, 4 * ncols), dpi=300, sharex=True)
    axes2 = np.atleast_1d(axes2)

    marker_props = {
        'A': {'color_in': '#d62728', 'color_sat': '#d62728', 'mk_in': 'o', 'mk_sat': '^', 'label_in': 'In-situ (Day)', 'label_sat': 'AMSR2 (Day)'},
        'D': {'color_in': '#1f77b4', 'color_sat': '#1f77b4', 'mk_in': 'o', 'mk_sat': '^', 'label_in': 'In-situ (Night)', 'label_sat': 'AMSR2 (Night)'}
    }

    legend_handles = []
    
    for i, st_en in enumerate(ordered_stations_en):
        ax = axes2[i]
        st_cn = name_map_en2cn[st_en]
        st_pinyin = CONFIG['PINYIN_MAP'].get(st_cn, st_en)
        df_sub = df_ts[df_ts['station'] == st_en]

        for orbit in orbits_plot:
            df_orb = df_sub[df_sub['orbit'] == orbit]
            if df_orb.empty:
                continue
            
            props = marker_props[orbit]
            
            sc_in = ax.scatter(df_orb['datetime'], df_orb['station_val'], 
                               facecolors='none', edgecolors=props['color_in'], 
                               marker=props['mk_in'], s=60, linewidths=1.5, label=props['label_in'])
                               
            sc_sat = ax.scatter(df_orb['datetime'], df_orb['amsr2_val'], 
                                color=props['color_sat'], marker=props['mk_sat'], 
                                s=60, alpha=0.8, label=props['label_sat'])

            # 收集图例（仅在第一张子图且未添加过时添加）
            if i == 0:
                labels_existing = [h.get_label() for h in legend_handles]
                if props['label_in'] not in labels_existing:
                    legend_handles.extend([sc_in, sc_sat])

        ax.set_title(f"({chr(97 + i)}) {st_pinyin}", loc='left', fontsize=15)
        ax.set_ylabel("LST (K)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', labelsize=12)

    axes2[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes2[-1].set_xlabel("Date", fontsize=14)

    fig2.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=13, frameon=False)
    
    plt.tight_layout()
    timeseries_save_path = os.path.join(CONFIG['SAVE_DIR'], 'LST_Validation_TimeSeries.jpg')
    fig2.savefig(timeseries_save_path, dpi=300, bbox_inches='tight')
    print(f"  时间序列图已导出至: {timeseries_save_path}")
    plt.show()

    print("\n模型验证流程结束。")

if __name__ == '__main__':
    main()
