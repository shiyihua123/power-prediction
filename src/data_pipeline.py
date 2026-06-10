import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import timedelta

def load_single_series(
    filepath,
    time_col='date',
    value_col='value',
    tz='UTC',
    fill_method=None
):
    """
    读取单个两列 CSV 文件，处理时区、缺失值，返回 pd.DataFrame
    - tz: 目标时区，原始数据假定为无时区的 UTC，先 localize 再 convert
    - fill_method: 'ffill', 'bfill', 'linear', 'zero', 'mean', None(丢弃)
    - 返回: DataFrame，包含时间列和值列，时间不作为索引
    """
    df = pd.read_csv(filepath)
    
    # 列检查
    missing_cols = [col for col in [time_col, value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"文件 {filepath} 缺少列: {missing_cols}。现有: {list(df.columns)}")
    
    # 时间解析与时区处理（假设原始为 UTC）
    df[time_col] = pd.to_datetime(df[time_col])
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize('UTC').dt.tz_convert(tz)
    else:
        # 如果已有时区，直接转换到目标时区
        df[time_col] = df[time_col].dt.tz_convert(tz)
    
    # 去重排序
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    
    # 缺失值处理
    missing_count = df[value_col].isnull().sum()
    if missing_count > 0:
        if fill_method is None:
            print(f"警告: {filepath} 含 {missing_count} 个缺失值，已丢弃。")
            df = df.dropna(subset=[value_col]).reset_index(drop=True)
        elif fill_method == 'ffill':
            df[value_col] = df[value_col].ffill()
            print(f"信息: {filepath} 含 {missing_count} 个缺失值，已用向前填充。")
        elif fill_method == 'bfill':
            df[value_col] = df[value_col].bfill()
            print(f"信息: {filepath} 含 {missing_count} 个缺失值，已用向后填充。")
        elif fill_method == 'linear':
            df[value_col] = df[value_col].interpolate(method='linear')
            print(f"信息: {filepath} 含 {missing_count} 个缺失值，已用线性插值。")
        elif fill_method == 'zero':
            df[value_col] = df[value_col].fillna(0)
            print(f"信息: {filepath} 含 {missing_count} 个缺失值，已用0填充。")
        elif fill_method == 'mean':
            df[value_col] = df[value_col].fillna(df[value_col].mean())
            print(f"信息: {filepath} 含 {missing_count} 个缺失值，已用均值填充。")
        else:
            raise ValueError(f"不支持的 fill_method: {fill_method}。")
    
    # 重命名值列为文件名（不含扩展名）
    df = df.rename(columns={value_col: Path(filepath).stem})
    return df

def interpolate_data():
    pass

def load_all_series(file_dict, time_col='date', value_col='value',
                    tz='UTC', fill_method=None, target_freq='h'):
    """
    批量加载并时间对齐多个 CSV 文件，返回宽表 DataFrame
    - file_dict: 字典，{ 列名: 文件路径 }，如 {'price': 'rawData/price.csv'}
    - target_freq: 目标频率，如 'h'、'15min'、'D'
    - 返回: 对齐后的宽表，列名为给定的字典键，时间不作为索引
    """
    if not file_dict:
        raise ValueError("file_dict 为空，请提供至少一个文件。")

    print(f"开始加载 {len(file_dict)} 个文件...")

    df_list = []
    for col_name, fpath in file_dict.items():
        df = load_single_series(fpath, time_col=time_col, value_col=value_col,
                               tz=tz, fill_method=fill_method)
        # 重命名值列为字典键指定的名称
        df = df.rename(columns={Path(fpath).stem: col_name})
        df_list.append(df)
        print(f"  ✓ {col_name} <- {Path(fpath).name}  {len(df)} 条，{df[time_col].min()} ~ {df[time_col].max()}")

    # 对齐：按时间列合并所有 DataFrame
    aligned = df_list[0]
    for df in df_list[1:]:
        aligned = pd.merge(aligned, df, on=time_col, how='inner')

    if aligned.empty:
        raise ValueError("共同时间点交集为空。")

    print(f"\n时间交集: {len(aligned)} 个时间点，{aligned[time_col].min()} ~ {aligned[time_col].max()}")

    # 如果指定了目标频率，进行重采样
    if target_freq:
        # 临时设置时间列为索引进行重采样
        aligned_resampled = aligned.set_index(time_col).resample(target_freq).mean().reset_index()
        
        # 统计重采样后产生的缺失值
        nan_count = aligned_resampled.drop(columns=[time_col]).isnull().sum().sum()
        if nan_count > 0:
            # 获取有缺失值的时间点
            nan_rows = aligned_resampled[aligned_resampled.drop(columns=[time_col]).isnull().any(axis=1)]
            nan_times = nan_rows[time_col].tolist()
            print(f"重采样产生 {nan_count} 个缺失值，缺失时间点:")
            for t in nan_times:
                print(f"  - {t}")
            # raise ValueError("backcast曲线未实现")
            aligned_resampled = aligned_resampled.set_index(time_col).interpolate(method='linear').reset_index()
            # 进行断点时间点数据填充，用backcast曲线
            # interpolate_data(aligned_resampled)
        
        print(f"已重采样到 '{target_freq}'，现有 {len(aligned_resampled)} 个时间点。")
        return aligned_resampled

    aligned = aligned.sort_values(time_col).reset_index(drop=True)
    return aligned


def add_time_feature(df, time_features, time_col='date'):
    # 这里进行特征工程，例如添加小时、星期几、月份等
    # 时间从列中提取，不再从索引中提取
    # 预计算常用时间分量
    hour = df[time_col].dt.hour
    weekday = df[time_col].dt.weekday  # 0=Mon, 6=Sun
    month = df[time_col].dt.month
    doy = df[time_col].dt.dayofyear
    local_time = df[time_col]  # 已经是本地时区

    for feat_name in time_features:
        if feat_name == 'hour_sin':
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        elif feat_name == 'hour_cos':
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        elif feat_name == 'weekday_sin':
            df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        elif feat_name == 'weekday_cos':
            df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        elif feat_name == 'month_sin':
            df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        elif feat_name == 'month_cos':
            df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        elif feat_name == 'hour':
            df['hour'] = hour
        elif feat_name == 'weekday':
            df['weekday'] = weekday
        elif feat_name == 'month':
            df['month'] = month

        # 一周内第几个小时：168小时周期
        elif feat_name == 'how_sin':
            hour_of_week = weekday * 24 + hour
            df['how_sin'] = np.sin(2 * np.pi * hour_of_week / 168)
        elif feat_name == 'how_cos':
            hour_of_week = weekday * 24 + hour
            df['how_cos'] = np.cos(2 * np.pi * hour_of_week / 168)

        # 年内日周期
        elif feat_name == 'doy_sin':
            df['doy_sin'] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        elif feat_name == 'doy_cos':
            df['doy_cos'] = np.cos(2 * np.pi * (doy - 1) / 365.25)

        # 布尔特征
        elif feat_name == 'is_weekend':
            df['is_weekend'] = (weekday >= 5).astype(float)
        elif feat_name == 'is_workday':
            df['is_workday'] = (weekday < 5).astype(float)

        # 峰谷时段
        elif feat_name == 'is_night':
            df['is_night'] = ((hour >= 0) & (hour <= 5)).astype(float)
        elif feat_name == 'is_morning_peak':
            df['is_morning_peak'] = ((hour >= 7) & (hour <= 10) & (weekday < 5)).astype(float)
        elif feat_name == 'is_evening_peak':
            df['is_evening_peak'] = ((hour >= 17) & (hour <= 20) & (weekday < 5)).astype(float)
        elif feat_name == 'is_business_hour':
            df['is_business_hour'] = ((hour >= 8) & (hour <= 18) & (weekday < 5)).astype(float)
        elif feat_name == 'is_offpeak':
            df['is_offpeak'] = ((hour <= 6) | (hour >= 22)).astype(float)

        # 季节特征
        elif feat_name == 'is_winter':
            df['is_winter'] = local_time.dt.month.isin([12, 1, 2]).astype(float)
        elif feat_name == 'is_summer':
            df['is_summer'] = local_time.dt.month.isin([6, 7, 8]).astype(float)
        elif feat_name == 'is_heating_season':
            df['is_heating_season'] = local_time.dt.month.isin([10, 11, 12, 1, 2, 3]).astype(float)

        # UTC 偏移量
        elif feat_name == 'utc_offset':
            df['utc_offset'] = local_time.map(
                lambda x: x.utcoffset().total_seconds() / 3600.0
            ).astype(float)
        elif feat_name == 'is_dst':
            df['is_dst'] = local_time.map(
                lambda x: 1.0 if x.dst() and x.dst().total_seconds() != 0 else 0.0
            )

    return df

def add_features(df, config):
    local_tz = config['data']['feature_kwargs']['local_tz']
    time_features = config['data']['feature_kwargs'].get('time_features', []) or []
    delta_features = config['data']['feature_kwargs'].get('delta_features', []) or []
    time_col = config['data']['raw_col'][0]

    df_feat = df.copy()
    # 转为本地时区（操作时间列而非索引）
    df_feat[time_col] = df_feat[time_col].dt.tz_convert(local_tz)
    # 添加时间特征
    df_feat = add_time_feature(df_feat, time_features, time_col)
    # 计算价格差分特征
    for feat_name in delta_features:
        if feat_name == 'price_delta_SE3_SE2':
            df_feat[feat_name] = df_feat['price_SE3'] - df_feat['price_SE2']
        elif feat_name == 'price_delta_SE4_SE2':
            df_feat[feat_name] = df_feat['price_SE4'] - df_feat['price_SE2']

    
    # 转回 UTC
    df_feat[time_col] = df_feat[time_col].dt.tz_convert('UTC')
    
    return df_feat

def prepare_data(config):
    """根据配置文件加载数据、划分训练/验证/测试集，并生成特征。
    返回: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # 1. 加载配置
    data_cfg = config['data']
    feature_kwargs = config['data']['feature_kwargs']
    time_col = data_cfg['raw_col'][0]  # 时间列名
    
    # 2. 加载并时间对齐所有文件
    df = load_all_series(
        file_dict=data_cfg['files'],
        time_col=time_col,           # 时间列名
        value_col=data_cfg['raw_col'][1],         # 值列名
        tz='UTC',
        fill_method=None,          # 暂时保持默认丢弃缺失，后面可以改为从配置读
        target_freq=data_cfg['freq']
    )
    print(f"总数据形状: {df.shape}")
    
    # 添加时间列参数到 feature_kwargs
    df_full = add_features(df, config)

    return df_full


def build_future_df(df_full, config):
    """
    根据现有数据和配置构造未来预测所需的 DataFrame
    - df_full: 包含特征的完整数据
    - config: 配置字典
    - 返回: 未来预测窗口的 DataFrame，包含时间列和时间特征
    """
    # 提取配置参数
    time_col = config['data']['raw_col'][0]
    prediction_window = config['data']['prediction_window']
    insured_date_str = config['data']['insured_date']
    if insured_date_str == "auto":
        insured_date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    insured_time = config['data']['insured_time']
    freq = config['data']['freq']
    local_tz = config['data']['feature_kwargs']['local_tz']

    # 获取数据最后时间点
    last_time = df_full[time_col].iloc[-1]
    
    # 将字符串日期转换为 datetime 对象（UTC）
    insured_date_utc = pd.to_datetime(insured_date_str).tz_localize('UTC')
    
    # 根据 insured_time 构造本地时间点（支持 0-23 小时）
    insured_date_local = insured_date_utc.tz_convert(local_tz)
    
    # 将 insured_time 转换为整数
    try:
        target_hour = int(insured_time)
    except ValueError:
        raise ValueError(f"insured_time 必须是整数字符串，当前值: {insured_time}")
    
    # 验证小时范围
    if not (0 <= target_hour <= 23):
        raise ValueError(f"insured_time 必须是 0-23 之间的整数，当前值: {target_hour}")
    
    # 构造目标时间点
    insured_date_local_target = insured_date_local.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if target_hour == 0:
        forecast_date = insured_date_local_target.tz_convert('UTC') + timedelta(days=1)
    elif target_hour == 12:
        forecast_date = insured_date_local_target.tz_convert('UTC') + timedelta(days=2)
    

    
    # 计算从 last_time 到 forecast_date 之间的时间点数
    forecast_index = pd.date_range(start=last_time, end=forecast_date, freq=freq, inclusive='right')
    horizon_total = len(forecast_index) + prediction_window - 1
    
    print(f"发布时间: {insured_date_local_target.tz_convert('UTC')}")
    print(f"预测时间点数量: {horizon_total}")
    
    # 生成未来时间序列：从 last_time 开始，共 horizon_total 个时间点
    future_index = pd.date_range(start=last_time, periods=horizon_total + 1, freq=freq)[1:]
    
    # 创建未来 df，包含时间列
    future_df = pd.DataFrame({time_col: future_index})
    
    # 添加时间特征到未来 df
    local_tz = config['data']['feature_kwargs']['local_tz']
    time_features = config['data']['feature_kwargs'].get('time_features', []) or []
    
    # 转为本地时区（操作时间列而非索引）
    future_df[time_col] = future_df[time_col].dt.tz_convert(local_tz)
    future_df = add_time_feature(future_df, time_features, time_col)
    
    future_df[time_col] = future_df[time_col].dt.tz_convert('UTC')
    
    return future_df, horizon_total
