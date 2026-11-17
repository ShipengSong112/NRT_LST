import os
import glob
import re
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import warnings
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

warnings.filterwarnings("ignore")

# ================= Configuration =================

# --- 1. Path Configuration ---
MODIS_DIR = r'E:\LST\MYD11A1\CHINA_mean'
# 已自动切换至夜晚/降轨对应的 D 文件夹
AMSR2_DIR = r'E:\E_LST\result\Surface_Retrieval_Robust_ROI_10channel_lst_only\D'
ERA5_ROOT_DIR = r'G:\ERA5\LST' 
save_name = r'E:\E_LST\JPG\MODIS_AMSR2_ERA5_Compare_Night.jpg'

# --- 2. Parameter Configuration ---
TARGET_DATES = ['20200401', '20200701', '20201001', '20201201']
EXTENT = [70, 140, 15, 55] 

LST_VMIN = 250
LST_VMAX = 330
LST_CMAP = 'RdYlBu_r'
MODIS_SCALE_FACTOR = 1.0  

# --- 3. Orbit Selection ---
# 切换为降轨夜晚模式
ORBIT_MODE = 'Descending' 

if ORBIT_MODE == 'Ascending':
    AMSR2_TAG = 'A'  
    MODIS_BAND_IDX = 1
    TITLE_SUFFIX = 'Day'
else:
    AMSR2_TAG = 'D'  
    MODIS_BAND_IDX = 2
    TITLE_SUFFIX = 'Night'

# ================= Utility Functions =================

def filter_amsr2_nc_files(file_list, target_date, orbit_tag):
    """Filter AMSR2 NetCDF files for the target date and orbit."""
    selected = []
    for f in file_list:
        base = os.path.basename(f)
        if f.lower().endswith('.nc') and f'_{target_date}' in base and orbit_tag in base:
            selected.append(f)
    return sorted(selected)

def get_modis_file(date_str):
    fname = f"MYD11A1_{date_str}_QC_Mean.tif" 
    return os.path.join(MODIS_DIR, fname)

def get_era5_file(date_str):
    year = date_str[:4]
    fname = f"ERA5_{date_str}_land.nc"
    return os.path.join(ERA5_ROOT_DIR, year, fname)

def get_amsr2_time_exact(filename):
    """Extract exact hour and minute from AMSR2 nc filename."""
    try:
        basename = os.path.basename(filename)
        match = re.search(r'GW1AM2_(\d{8})(\d{4})', basename)
        if match:
            time_str = match.group(2)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return hour, minute
    except:
        return -1, -1
    return -1, -1

def get_era5_interpolated(era5_ds, target_hour, target_minute):
    """Perform temporal linear interpolation on ERA5 data."""
    weight = target_minute / 60.0
    idx_floor = target_hour
    idx_ceil = target_hour + 1
    
    if idx_floor >= 23:
        slice_interp = era5_ds['skt'].isel(valid_time=23)
    else:
        data_floor = era5_ds['skt'].isel(valid_time=idx_floor)
        data_ceil  = era5_ds['skt'].isel(valid_time=idx_ceil)
        slice_interp = data_floor * (1.0 - weight) + data_ceil * weight
        
    return slice_interp

def add_aligned_colorbar(fig, ax_target, mappable, label):
    pos = ax_target.get_position()
    cax_rect = [pos.x1 + 0.01, pos.y0, 0.015, pos.height]
    cax = fig.add_axes(cax_rect)
    cb = plt.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=14, fontname='Times New Roman')
    cb.ax.tick_params(labelsize=12)
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    return cb

def add_subplot_label(ax, label):
    ax.text(0.92, 0.08, label, transform=ax.transAxes, 
            fontsize=18, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9),
            fontname='Times New Roman', zorder=200)

def add_map_features(ax, show_left_label=False, show_bottom_label=False):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    
    # 强制所有边框显示在最顶层 (zorder=1000)，防止被数据层遮挡
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
        spine.set_zorder(1000)

    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='white')
    ax.add_feature(ocean, zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=101, color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=101)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=102)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_label
    gl.bottom_labels = show_bottom_label
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 11, 'color': 'black', 'family': 'Times New Roman'}
    gl.ylabel_style = {'size': 11, 'color': 'black', 'family': 'Times New Roman'}

# ================= Main Plotting Logic =================

def plot_three_columns_comparison():
    amsr2_files_all = glob.glob(os.path.join(AMSR2_DIR, "*.nc"))
    
    fig = plt.figure(figsize=(16, 12))
    
    # SCI Style Font Configuration
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False 

    # 4 rows x 3 columns layout
    gs = gridspec.GridSpec(4, 3, figure=fig, 
                           width_ratios=[1, 1, 1], 
                           wspace=0.05, hspace=0.08,
                           left=0.05, right=0.92, bottom=0.05, top=0.92)
    
    # Generate labels (a) to (l)
    labels_list = [f"({chr(97+i)})" for i in range(12)] 

    for i, date_str in enumerate(TARGET_DATES):
        if i >= 4: break
        
        # Initialize subplots
        ax_modis = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax_amsr2 = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        ax_era5  = fig.add_subplot(gs[i, 2], projection=ccrs.PlateCarree())
        
        is_last_row = (i == 3)
        
        add_map_features(ax_modis, show_left_label=True, show_bottom_label=is_last_row)
        add_map_features(ax_amsr2, show_left_label=False, show_bottom_label=is_last_row)
        add_map_features(ax_era5, show_left_label=False, show_bottom_label=is_last_row)
        
        mesh_ref = None # Reference for Colorbar
        
        # ---------------- 1. MODIS ----------------
        modis_path = get_modis_file(date_str)
        if os.path.exists(modis_path):
            try:
                da = rioxarray.open_rasterio(modis_path, masked=True)
                raw = da[MODIS_BAND_IDX - 1] if da.shape[0] >= MODIS_BAND_IDX else da[0]
                da_modis_data = raw * MODIS_SCALE_FACTOR
                ax_modis.pcolormesh(da.x, da.y, da_modis_data, 
                                    transform=ccrs.PlateCarree(),
                                    cmap=LST_CMAP, vmin=LST_VMIN, vmax=LST_VMAX, zorder=10)
            except Exception as e:
                print(f"[{date_str}] MODIS Load Error: {e}")
        
        # ---------------- 2. ERA5 Preparation ----------------
        era5_path = get_era5_file(date_str)
        era5_ds = None
        if os.path.exists(era5_path):
            try:
                era5_ds = xr.open_dataset(era5_path)
            except Exception as e:
                print(f"[{date_str}] ERA5 Load Error: {e}")

        # ---------------- 3. AMSR2 & ERA5 Interpolation ----------------
        day_files = filter_amsr2_nc_files(amsr2_files_all, date_str, AMSR2_TAG)
        
        if day_files:
            for f_path in day_files:
                try:
                    # Read AMSR2 nc file
                    amsr2_ds = xr.open_dataset(f_path)
                    lat = amsr2_ds['lat'].values
                    lon = amsr2_ds['lon'].values
                    #lst_data = amsr2_ds['Retrieved_T_skin(K)'].values
                    lst_data = amsr2_ds['Retrieved_T_skin_K'].values
                    # Plot AMSR2
                    mesh_ref = ax_amsr2.pcolormesh(lon, lat, lst_data, 
                                        transform=ccrs.PlateCarree(),
                                        cmap=LST_CMAP, vmin=LST_VMIN, vmax=LST_VMAX, zorder=10)
                    
                    # Plot Interpolated ERA5
                    if era5_ds is not None:
                        h, m = get_amsr2_time_exact(f_path)
                        if h != -1:
                            era5_interp = get_era5_interpolated(era5_ds, h, m)
                            lon_coord = 'longitude' if 'longitude' in era5_interp.coords else 'lon'
                            lat_coord = 'latitude' if 'latitude' in era5_interp.coords else 'lat'
                            
                            ax_era5.pcolormesh(era5_interp[lon_coord], era5_interp[lat_coord], era5_interp.values,
                                               transform=ccrs.PlateCarree(),
                                               cmap=LST_CMAP, vmin=LST_VMIN, vmax=LST_VMAX, zorder=10)
                    
                    amsr2_ds.close()
                except Exception as e:
                    print(f"Error processing {os.path.basename(f_path)}: {e}")
        else:
            print(f"[{date_str}] No AMSR2 files found for orbit {AMSR2_TAG}.")

        # ---------------- Set Titles & Labels ----------------
        if i == 0:
            ax_modis.set_title(f"MODIS ({TITLE_SUFFIX})", fontsize=15, fontweight='bold', fontname='Times New Roman')
            ax_amsr2.set_title(f"AMSR2 ({TITLE_SUFFIX})", fontsize=15, fontweight='bold', fontname='Times New Roman')
            ax_era5.set_title("ERA5 (Interpolated to AMSR2 Overpass Time)", fontsize=15, fontweight='bold', fontname='Times New Roman')

        # Arrange labels by column:
        add_subplot_label(ax_modis, labels_list[i])
        add_subplot_label(ax_amsr2, labels_list[i + 4])
        add_subplot_label(ax_era5, labels_list[i + 8])

        # Add colorbar on the right of the third column (ERA5)
        if mesh_ref:
            add_aligned_colorbar(fig, ax_era5, mesh_ref, 'LST (K)')

    if era5_ds: 
        era5_ds.close()
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully: {save_name}")
    plt.show()

if __name__ == "__main__":
    plot_three_columns_comparison()