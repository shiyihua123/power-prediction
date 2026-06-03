import random
import numpy as np
import yaml
from src.data_pipeline import prepare_data, build_future_df
from src.models.model_registry import get_model
import pandas as pd

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']

df_full = prepare_data(config)
# _, horizon_total = build_future_df(df_full, config)
config['horizon_total'] = config['data']['prediction_window'] + 24
model = get_model(model_name, config)

cv_results = model.cross_validate(df_full)

# 保存cv_results为csv
output_path = f'outputs/{model_name}/cv_results_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
import os
os.makedirs(f'outputs/{model_name}', exist_ok=True)
cv_results.to_csv(output_path, index=False)

print(f"cv_results结果已保存到 {output_path}")
