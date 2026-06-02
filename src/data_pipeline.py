import pandas as pd
import numpy as np

def load_demo_data(start_date='2026-01-01', periods=200, freq='h'):
    """生成演示用的模拟电力数据"""
    time_index = pd.date_range(start=start_date, periods=periods, freq=freq)
    # 模拟电力数据：基础值 + 日周期 + 噪声
    base = 100
    hour = time_index.hour
    daily_pattern = 20 * np.sin(2 * np.pi * hour / 24)
    noise = np.random.normal(0, 5, size=periods)
    values = base + daily_pattern + noise
    y = pd.Series(values, index=time_index, name='power')
    # 构造特征表（简单带滞后特征）
    X = pd.DataFrame(index=time_index)
    X['hour'] = hour
    X['lag_1'] = y.shift(1)
    X['lag_24'] = y.shift(24)
    X = X.dropna()
    y = y[X.index]
    return X, y