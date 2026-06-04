"""模型预测脚本。"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import yaml
import pandas as pd
from src.models.model_registry import get_model
from src.data_pipeline import prepare_data, build_future_df

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('model_config.yaml') as f:
    model_config = yaml.safe_load(f)
    
model_name = config['model_name']

df_full = prepare_data(config)
future_df, horizon_total = build_future_df(df_full, config)
config['horizon_total'] = horizon_total

model = get_model(model_name, config, model_config).load(f"models_results/{model_name}/model.pkl", config, model_config)

pred_df = model.predict(future_df)
prediction_window = config['data']['prediction_window']
pred_df = pred_df.tail(prediction_window)
output_path = f'outputs/{model_name}/prediction_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
os.makedirs(f'outputs/{model_name}', exist_ok=True)
pred_df.to_csv(output_path, index=False)

print(f"预测结果已保存到 {output_path}")
