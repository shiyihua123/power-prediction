"""模型训练脚本。"""
import os
import sys

# 确保项目根目录在 sys.path 中（无论从哪里执行、用什么环境）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
import yaml
from src.data_pipeline import prepare_data, build_future_df
from src.models.model_registry import get_model
import src.models.neuralforecast_model  # 触发 @register_model 装饰器

with open('config.yaml') as f:
    config = yaml.safe_load(f)

name = config['model']['name']

df_full = prepare_data(config)
_, horizon_total = build_future_df(df_full, config)
config['horizon_total'] = horizon_total
model = get_model(name, config)
model.fit(df_full)

os.makedirs(f"models_results/{name}", exist_ok=True)
model.save(f"models_results/{name}/model.pkl")
print(f"模型 {name} 训练完成，已保存到 models_results/{name}/model.pkl")
