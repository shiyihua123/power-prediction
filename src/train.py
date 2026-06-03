import random
import numpy as np
import yaml
from data_pipeline import prepare_data, build_future_df
from models.model_registry import get_model

with open('config.yaml') as f:
    config = yaml.safe_load(f)

name = config['model']['name']

df_full = prepare_data(config)
_, horizon_total = build_future_df(df_full, config)
config['horizon_total'] = horizon_total
model = get_model(name, config)
model.fit(df_full)

import os
os.makedirs(f"models/{name}", exist_ok=True)

# 4. 保存模型
model.save(f"models/{name}/model.pkl")
print(f"模型 {name} 训练完成，已保存到 models/{name}/model.pkl")
