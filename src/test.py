"""交叉验证测试脚本。"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
import yaml
from src.data_pipeline import prepare_data, build_future_df
from src.models.model_registry import get_model
import pandas as pd

with open('config.yaml') as f:
    config = yaml.safe_load(f)

with open('model_config.yaml') as f:
    model_config = yaml.safe_load(f)

model_name = config['model_name']

df_full = prepare_data(config)
config['horizon_total'] = config['data']['prediction_window'] + 24
model = get_model(model_name, config, model_config)

cv_results = model.cross_validate(df_full)

output_path = f'outputs/{model_name}/cv/cv_results_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
os.makedirs(f'outputs/{model_name}/cv', exist_ok=True)
cv_results.to_csv(output_path, index=False)

print(f"cv_results结果已保存到 {output_path}")
