import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results_from_npz(file_path):
    """Load results from npz file directly"""
    try:
        data = np.load(file_path, allow_pickle=True)

        # Check if data contains a 'results' key which holds the actual data
        if "results" in data:
            return data["results"]

        # Otherwise, get the first array
        keys = list(data.keys())
        if keys:
            first_key = keys[0]
            if isinstance(data[first_key], np.ndarray) and data[
                first_key
            ].dtype == np.dtype("O"):
                return data[first_key]

        # As a fallback, create a dict from all keys
        return {k: data[k] for k in data}
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        # 返回一个模拟的空数组
        return np.array({})

def Read_Galaxy(galaxy_name):
    """读取星系的三种分析数据"""
    def Read_otp(galaxy_name, mode_name="P2P"):
        """Read output file for a specific mode"""
        file_path = (
            "./output/"
            + galaxy_name
            + "/"
            + galaxy_name
            + "_stack/Data/"
            + galaxy_name
            + "_stack_"
            + mode_name
            + "_results.npz"
        )
        try:
            if os.path.exists(file_path):
                df = load_results_from_npz(file_path)
                return df
            else:
                logger.warning(f"文件不存在: {file_path}")
                return np.array({})
        except Exception as e:
            logger.error(f"Error reading {mode_name} data for {galaxy_name}: {e}")
            return np.array({})
    
    # 读取原始数据
    try:
        df_1 = Read_otp(galaxy_name)
        df_2 = Read_otp(galaxy_name, 'VNB')
        df_3 = Read_otp(galaxy_name, 'RDB')
        return df_1, df_2, df_3
    except Exception as e:
        logger.error(f"Error reading galaxy data: {e}")
        return np.array({}), np.array({}), np.array({})

def extract_galaxy_indices(p2p_data, vnb_data, rdb_data, snr_threshold=0, indices_names=None):
    """从三种模式的数据中提取光谱指数和R、Age、MOH等物理参数"""
    if indices_names is None:
        indices_names = ['Fe5015', 'Mgb']  # 默认只包含Fe5015和Mgb，避免Hbeta问题
    
    # 初始化结果字典
    result = {
        'p2p': {'indices': {}, 'metadata': {}},
        'vnb': {'indices': {}, 'metadata': {}},
        'rdb': {'indices': {}, 'metadata': {}}
    }
    
    # 处理P2P数据
    if p2p_data is not None and len(p2p_data) > 0:
        # 提取距离/半径
        if 'distance' in p2p_data and p2p_data['distance'] is not None:
            try:
                distance_item = p2p_data['distance'].item() if hasattr(p2p_data['distance'], 'item') else p2p_data['distance']
                if 'field' in distance_item:
                    radius = distance_item['field']
                    result['p2p']['metadata']['R'] = radius.flatten() if hasattr(radius, 'flatten') else radius
            except Exception as e:
                logger.error(f"提取P2P半径时出错: {e}")
        
        # 提取信噪比
        if 'signal_noise' in p2p_data and p2p_data['signal_noise'] is not None:
            try:
                sn_item = p2p_data['signal_noise'].item() if hasattr(p2p_data['signal_noise'], 'item') else p2p_data['signal_noise']
                if 'snr' in sn_item:
                    snr = sn_item['snr']
                    result['p2p']['metadata']['SNR'] = snr.flatten() if hasattr(snr, 'flatten') else snr
            except Exception as e:
                logger.error(f"提取P2P信噪比时出错: {e}")
        
        # 提取光谱指数
        if 'indices' in p2p_data and p2p_data['indices'] is not None:
            try:
                indices_item = p2p_data['indices'].item() if hasattr(p2p_data['indices'], 'item') else p2p_data['indices']
                for index_name in indices_names:
                    if index_name in indices_item:
                        index_data = indices_item[index_name]
                        result['p2p']['indices'][index_name] = index_data.flatten() if hasattr(index_data, 'flatten') else index_data
            except Exception as e:
                logger.error(f"提取P2P光谱指数时出错: {e}")
        
        # 提取年龄数据
        if 'stellar_population' in p2p_data and p2p_data['stellar_population'] is not None:
            try:
                stellar_pop = p2p_data['stellar_population'].item() if hasattr(p2p_data['stellar_population'], 'item') else p2p_data['stellar_population']
                if 'age' in stellar_pop:
                    age = stellar_pop['age']
                    result['p2p']['metadata']['Age'] = age.flatten() if hasattr(age, 'flatten') else age
                if 'metallicity' in stellar_pop:
                    metallicity = stellar_pop['metallicity']
                    result['p2p']['metadata']['MOH'] = metallicity.flatten() if hasattr(metallicity, 'flatten') else metallicity
            except Exception as e:
                logger.error(f"提取P2P恒星年龄和金属丰度时出错: {e}")
    
    # 处理VNB数据
    if vnb_data is not None and len(vnb_data) > 0:
        # 提取距离/半径
        if 'distance' in vnb_data and vnb_data['distance'] is not None:
            try:
                distance_item = vnb_data['distance'].item() if hasattr(vnb_data['distance'], 'item') else vnb_data['distance']
                if 'bin_distances' in distance_item:
                    result['vnb']['metadata']['R'] = distance_item['bin_distances']
            except Exception as e:
                logger.error(f"提取VNB半径时出错: {e}")
        
        # 提取信噪比
        if 'snr' in vnb_data and vnb_data['snr'] is not None:
            try:
                result['vnb']['metadata']['SNR'] = vnb_data['snr']
            except Exception as e:
                logger.error(f"提取VNB信噪比时出错: {e}")
        
        # 提取光谱指数
        if 'bin_indices' in vnb_data and vnb_data['bin_indices'] is not None:
            try:
                bin_indices = vnb_data['bin_indices'].item() if hasattr(vnb_data['bin_indices'], 'item') else vnb_data['bin_indices']
                if 'bin_indices' in bin_indices:
                    for index_name in indices_names:
                        if index_name in bin_indices['bin_indices']:
                            result['vnb']['indices'][index_name] = bin_indices['bin_indices'][index_name]
            except Exception as e:
                logger.error(f"提取VNB光谱指数时出错: {e}")
                # 尝试备选路径
                if isinstance(vnb_data['bin_indices'], dict) and 'bin_indices' in vnb_data['bin_indices']:
                    for index_name in indices_names:
                        if index_name in vnb_data['bin_indices']['bin_indices']:
                            result['vnb']['indices'][index_name] = vnb_data['bin_indices']['bin_indices'][index_name]
        
        # 提取年龄和金属丰度数据
        if 'stellar_population' in vnb_data and vnb_data['stellar_population'] is not None:
            try:
                stellar_pop = vnb_data['stellar_population'].item() if hasattr(vnb_data['stellar_population'], 'item') else vnb_data['stellar_population']
                if 'age' in stellar_pop:
                    result['vnb']['metadata']['Age'] = stellar_pop['age']
                if 'metallicity' in stellar_pop:
                    result['vnb']['metadata']['MOH'] = stellar_pop['metallicity']
            except Exception as e:
                logger.error(f"提取VNB恒星年龄和金属丰度时出错: {e}")
    
    # 处理RDB数据
    if rdb_data is not None and len(rdb_data) > 0:
        # 提取距离/半径
        if 'distance' in rdb_data and rdb_data['distance'] is not None:
            try:
                distance_item = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
                if 'bin_distances' in distance_item:
                    result['rdb']['metadata']['R'] = distance_item['bin_distances']
            except Exception as e:
                logger.error(f"提取RDB半径时出错: {e}")
        
        # 提取信噪比
        if 'snr' in rdb_data and rdb_data['snr'] is not None:
            try:
                result['rdb']['metadata']['SNR'] = rdb_data['snr']
            except Exception as e:
                logger.error(f"提取RDB信噪比时出错: {e}")
        
        # 提取光谱指数
        if 'indices' in rdb_data and rdb_data['indices'] is not None:
            try:
                indices_item = rdb_data['indices'].item() if hasattr(rdb_data['indices'], 'item') else rdb_data['indices']
                for index_name in indices_names:
                    if index_name in indices_item:
                        result['rdb']['indices'][index_name] = indices_item[index_name]
            except Exception as e:
                logger.error(f"提取RDB光谱指数时出错: {e}")
                # 尝试备选路径
                if isinstance(rdb_data['indices'], dict):
                    for index_name in indices_names:
                        if index_name in rdb_data['indices']:
                            result['rdb']['indices'][index_name] = rdb_data['indices'][index_name]
        
        # 提取年龄和金属丰度数据
        if 'stellar_population' in rdb_data and rdb_data['stellar_population'] is not None:
            try:
                stellar_pop = rdb_data['stellar_population'].item() if hasattr(rdb_data['stellar_population'], 'item') else rdb_data['stellar_population']
                if 'age' in stellar_pop:
                    result['rdb']['metadata']['Age'] = stellar_pop['age']
                if 'metallicity' in stellar_pop:
                    result['rdb']['metadata']['MOH'] = stellar_pop['metallicity']
            except Exception as e:
                logger.error(f"提取RDB恒星年龄和金属丰度时出错: {e}")
    
    # 应用信噪比过滤
    if snr_threshold > 0:
        # P2P过滤
        if 'SNR' in result['p2p']['metadata']:
            snr = result['p2p']['metadata']['SNR']
            for index_name in result['p2p']['indices']:
                index_data = result['p2p']['indices'][index_name]
                if len(snr) == len(index_data):
                    mask = snr < snr_threshold
                    filtered_data = index_data.copy()
                    filtered_data[mask] = np.nan
                    result['p2p']['indices'][index_name] = filtered_data
            
            # 同样过滤元数据
            for meta_key in result['p2p']['metadata']:
                if meta_key != 'SNR':
                    meta_data = result['p2p']['metadata'][meta_key]
                    if len(snr) == len(meta_data):
                        mask = snr < snr_threshold
                        filtered_data = meta_data.copy()
                        filtered_data[mask] = np.nan
                        result['p2p']['metadata'][meta_key] = filtered_data
        
        # VNB过滤
        if 'SNR' in result['vnb']['metadata']:
            snr = result['vnb']['metadata']['SNR']
            for index_name in result['vnb']['indices']:
                index_data = result['vnb']['indices'][index_name]
                if len(snr) == len(index_data):
                    mask = snr < snr_threshold
                    filtered_data = index_data.copy()
                    filtered_data[mask] = np.nan
                    result['vnb']['indices'][index_name] = filtered_data
            
            # 同样过滤元数据
            for meta_key in result['vnb']['metadata']:
                if meta_key != 'SNR':
                    meta_data = result['vnb']['metadata'][meta_key]
                    if len(snr) == len(meta_data):
                        mask = snr < snr_threshold
                        filtered_data = meta_data.copy()
                        filtered_data[mask] = np.nan
                        result['vnb']['metadata'][meta_key] = filtered_data
        
        # RDB过滤
        if 'SNR' in result['rdb']['metadata']:
            snr = result['rdb']['metadata']['SNR']
            for index_name in result['rdb']['indices']:
                index_data = result['rdb']['indices'][index_name]
                if len(snr) == len(index_data):
                    mask = snr < snr_threshold
                    filtered_data = index_data.copy()
                    filtered_data[mask] = np.nan
                    result['rdb']['indices'][index_name] = filtered_data
            
            # 同样过滤元数据
            for meta_key in result['rdb']['metadata']:
                if meta_key != 'SNR':
                    meta_data = result['rdb']['metadata'][meta_key]
                    if len(snr) == len(meta_data):
                        mask = snr < snr_threshold
                        filtered_data = meta_data.copy()
                        filtered_data[mask] = np.nan
                        result['rdb']['metadata'][meta_key] = filtered_data
    
    return result

def load_model_data(data_file):
    """加载模型数据"""
    try:
        return pd.read_csv(data_file)
    except Exception as e:
        logger.error(f"加载模型数据出错: {e}")
        return None

def create_heatmap_background(ax, x_data, y_data, z_data, x_range, y_range, 
                             resolution=100, cmap='viridis', alpha=0.7):
    """
    在指定的坐标轴上创建热图背景
    
    参数:
    ---------
    ax : matplotlib.axes.Axes
        要绘制热图的轴对象
    x_data, y_data : array-like
        用于构建热图的x和y坐标数据
    z_data : array-like
        用于确定热图颜色的z数据
    x_range, y_range : tuple
        x和y轴的范围
    resolution : int, optional
        热图的分辨率，默认为100
    cmap : str, optional
        色彩映射名称，默认为'viridis'
    alpha : float, optional
        热图的透明度，默认为0.7
    
    返回:
    --------
    matplotlib.collections.QuadMesh
        热图对象，可用于添加颜色条
    """
    # 确保数据有效
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data) & ~np.isnan(z_data)
    x = np.array(x_data)[valid_mask]
    y = np.array(y_data)[valid_mask]
    z = np.array(z_data)[valid_mask]
    
    if len(x) < 10:
        logger.warning("数据点太少，无法创建热图背景")
        return None
    
    try:
        # 创建网格
        xi = np.linspace(x_range[0], x_range[1], resolution)
        yi = np.linspace(y_range[0], y_range[1], resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # 使用griddata进行插值
        zi = griddata((x, y), z, (xi, yi), method='cubic', fill_value=np.nan)
        
        # 创建热图
        heatmap = ax.pcolormesh(xi, yi, zi, cmap=cmap, alpha=alpha, shading='auto', zorder=0)
        
        # 绘制等高线
        contour = ax.contour(xi, yi, zi, colors='white', linewidths=0.5, alpha=0.4, zorder=1)
        
        return heatmap
    except Exception as e:
        logger.error(f"创建热图背景时出错: {e}")
        return None

def plot_galaxy_with_heatmap(galaxies, model_data_file, indices_pair=('Fe5015', 'Mgb'), 
                            color_vars=['R', 'Age', 'MOH'], snr_threshold=3.0, figsize=(18, 6),
                            dpi=300, age=1, save_path=None):
    """
    创建多个面板的光谱指数对比图，每个面板使用不同的物理参数作为热图背景
    
    参数:
    ---------
    galaxies : list of str
        要处理的星系名称列表
    model_data_file : str
        模型数据文件路径
    indices_pair : tuple, optional
        要绘制的光谱指数对，默认为('Fe5015', 'Mgb')
    color_vars : list, optional
        用作热图背景的变量列表，默认为['R', 'Age', 'MOH']
    snr_threshold : float, optional
        信噪比阈值，小于此值的数据点将被过滤
    figsize : tuple, optional
        图表大小，默认为(18, 6)
    dpi : int, optional
        图表分辨率，默认为300
    age : float, optional
        恒星演化模型的年龄，默认为1
    save_path : str, optional
        保存路径，如果不为None，图表将保存为PNG文件
    
    返回:
    -------
    tuple
        (fig, axes) - matplotlib的figure和axes对象
    """
    # 加载模型数据
    model_data = load_model_data(model_data_file)
    if model_data is None:
        logger.error("无法加载模型数据，退出")
        return None, None
    
    # 指定要作为X和Y的索引名称
    x_index, y_index = indices_pair
    
    # 准备数据存储
    data_points = {
        'p2p': {var: {'x': [], 'y': [], 'z': []} for var in color_vars},
        'vnb': {var: {'x': [], 'y': [], 'z': []} for var in color_vars},
        'rdb': {var: {'x': [], 'y': [], 'z': []} for var in color_vars}
    }
    
    # 处理所有星系数据
    for galaxy_name in galaxies:
        logger.info(f"处理星系: {galaxy_name}")
        
        try:
            # 读取星系数据
            p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
            
            # 提取指数数据和元数据
            galaxy_data = extract_galaxy_indices(p2p_data, vnb_data, rdb_data, snr_threshold, [x_index, y_index])
            
            # 收集每种解决方案的数据
            for mode in ['p2p', 'vnb', 'rdb']:
                if x_index in galaxy_data[mode]['indices'] and y_index in galaxy_data[mode]['indices']:
                    x_data = galaxy_data[mode]['indices'][x_index]
                    y_data = galaxy_data[mode]['indices'][y_index]
                    
                    # 对于每个颜色变量
                    for var in color_vars:
                        if var in galaxy_data[mode]['metadata']:
                            z_data = galaxy_data[mode]['metadata'][var]
                            
                            # 确保数据长度匹配
                            if len(x_data) == len(y_data) == len(z_data):
                                # 筛选有效数据点
                                valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data) & ~np.isnan(z_data)
                                if np.any(valid_mask):
                                    # 添加到数据集
                                    data_points[mode][var]['x'].extend(x_data[valid_mask])
                                    data_points[mode][var]['y'].extend(y_data[valid_mask])
                                    data_points[mode][var]['z'].extend(z_data[valid_mask])
        except Exception as e:
            logger.error(f"处理星系 {galaxy_name} 时出错: {e}")
            continue
    
    # 创建图表 - 每个颜色变量一个面板
    fig, axes = plt.subplots(1, len(color_vars), figsize=figsize, dpi=dpi)
    
    # 确保axes是数组
    if len(color_vars) == 1:
        axes = [axes]
    
    # 设置坐标范围
    x_values = []
    y_values = []
    for mode in ['p2p', 'vnb', 'rdb']:
        for var in color_vars:
            x_values.extend(data_points[mode][var]['x'])
            y_values.extend(data_points[mode][var]['y'])
    
    if len(x_values) > 0 and len(y_values) > 0:
        x_range = (min(x_values), max(x_values))
        y_range = (min(y_values), max(y_values))
    else:
        x_range = (1, 5)
        y_range = (1, 3)
    
    # 筛选指定年龄的模型数据
    age_data = model_data[model_data['Age'] == age].copy()
    
    # 对于每个颜色变量，创建一个面板
    for i, var in enumerate(color_vars):
        ax = axes[i]
        
        # 设置标题和标签
        ax.set_title(f'{x_index} vs {y_index} colored by {var}', fontsize=14)
        ax.set_xlabel(x_index, fontsize=12)
        ax.set_ylabel(y_index, fontsize=12)
        
        # 绘制模型网格
        # 获取独特的ZoH和AoFe值
        zoh_unique = sorted(age_data['ZoH'].unique())
        aofe_unique = sorted(age_data['AoFe'].unique())
        
        # 绘制ZoH网格线
        for zoh in zoh_unique:
            zoh_data = age_data[age_data['ZoH'] == zoh]
            ax.plot(zoh_data[x_index], zoh_data[y_index], '-', 
                   color='black', alpha=0.4, linewidth=1.0, zorder=3)
        
        # 绘制AoFe网格线
        for aofe in aofe_unique:
            aofe_data = age_data[age_data['AoFe'] == aofe]
            ax.plot(aofe_data[x_index], aofe_data[y_index], '--', 
                   color='black', alpha=0.4, linewidth=1.0, zorder=3)
        
        # 添加网格点标注
        for zoh in [-1.0, 0.0, 0.5]:
            if zoh in zoh_unique:
                for aofe in [0.0, 0.3, 0.5]:
                    if aofe in aofe_unique:
                        point_data = age_data[(age_data['ZoH'] == zoh) & (age_data['AoFe'] == aofe)]
                        if len(point_data) > 0:
                            x_val = point_data[x_index].values[0]
                            y_val = point_data[y_index].values[0]
                            
                            ax.scatter(x_val, y_val, color='black', s=20, zorder=4)
                            
                            # 只标注关键点
                            label = f'[Z/H]={zoh:.1f}\n[α/Fe]={aofe:.1f}'
                            ax.annotate(label, (x_val, y_val), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.8, zorder=4,
                                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        
        # 合并所有模式的数据
        all_x = []
        all_y = []
        all_z = []
        for mode in ['vnb', 'rdb']:  # 只使用VNB和RDB数据进行热图
            all_x.extend(data_points[mode][var]['x'])
            all_y.extend(data_points[mode][var]['y'])
            all_z.extend(data_points[mode][var]['z'])
        
        # 创建热图背景
        if len(all_x) > 10:
            heatmap = create_heatmap_background(
                ax, all_x, all_y, all_z, 
                x_range, y_range, 
                resolution=100, 
                cmap='viridis', 
                alpha=0.7
            )
            
            # 添加颜色条
            if heatmap is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(heatmap, cax=cax)
                cbar.set_label(var, fontsize=10)
        
        # 绘制P2P数据点
        if len(data_points['p2p'][var]['x']) > 0:
            ax.scatter(
                data_points['p2p'][var]['x'], 
                data_points['p2p'][var]['y'],
                c='gray', s=3, alpha=0.2, marker='.', 
                label=f'P2P ({len(data_points["p2p"][var]["x"])})', zorder=5
            )
        
        # 绘制VNB数据点
        if len(data_points['vnb'][var]['x']) > 0:
            ax.scatter(
                data_points['vnb'][var]['x'], 
                data_points['vnb'][var]['y'],
                c='blue', s=30, alpha=0.6, marker='+', linewidths=1.0,
                label=f'VNB ({len(data_points["vnb"][var]["x"])})', zorder=6
            )
        
        # 绘制RDB数据点
        if len(data_points['rdb'][var]['x']) > 0:
            ax.scatter(
                data_points['rdb'][var]['x'], 
                data_points['rdb'][var]['y'],
                c='red', s=30, alpha=0.6, marker='x', linewidths=1.0,
                label=f'RDB ({len(data_points["rdb"][var]["x"])})', zorder=7
            )
        
        # 添加图例
        ax.legend(loc='upper left', fontsize=9, framealpha=0.7)
        
        # 设置坐标轴范围
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # 设置刻度样式
        ax.tick_params(axis='both', which='both', labelsize=10, 
                      right=True, top=True, direction='in', width=1.0)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 添加总标题
    fig.suptitle(f'Combined Galaxies: {x_index} vs {y_index} (Age = {age} Gyr)', fontsize=16, y=0.98)
    
    # 添加包含的星系信息
    galaxy_text = "Galaxies included: " + ", ".join(galaxies[:5])
    if len(galaxies) > 5:
        galaxy_text += f" and {len(galaxies) - 5} others"
    plt.figtext(0.5, 0.01, galaxy_text, fontsize=10, ha='center')
    
    # 调整间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"已保存图表到: {save_path}")
    
    return fig, axes

def main():
    """主程序入口"""
    # 定义要处理的星系列表
    galaxies = [
        "VCC0308", "VCC0667", "VCC0990", "VCC1048", "VCC1154", 
        "VCC1193", "VCC1368", "VCC1410", "VCC1454", "VCC1499", 
        "VCC1549", "VCC1588", "VCC1695", "VCC1833", "VCC1896", 
        "VCC1902", "VCC1910", "VCC1949"
    ]
    
    # 设置信噪比阈值，过滤低信噪比数据
    snr_threshold = 3.0
    
    # 设置模型文件路径
    model_data_file = './TMB03/TMB03.csv'  # 请替换为实际路径
    
    # 创建带热图的多面板图
    fig, axes = plot_galaxy_with_heatmap(
        galaxies=galaxies,
        model_data_file=model_data_file,
        indices_pair=('Fe5015', 'Mgb'),  # 可以更改为其他指数对
        color_vars=['R', 'Age', 'MOH'],  # 用作热图背景的变量
        snr_threshold=snr_threshold,
        figsize=(18, 6),
        dpi=300,
        age=1,
        save_path="Galaxy_Heatmap_Background.png"
    )
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()