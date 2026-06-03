import yaml
import pandas as pd
from src.models.model_registry import get_model
from src.data_pipeline import prepare_data, build_future_df

# 1. 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
model_name = config['model']['name']

# 2. 准备数据
df_full = prepare_data(config)
future_df, horizon_total = build_future_df(df_full, config)
config['horizon_total'] = horizon_total

# 3. 加载模型（传入 config）
model = get_model(model_name, config).load(f"models/{model_name}/model.pkl", config)

# 4. 预测
pred_df = model.predict(future_df)
# 只保存后prediction_window 行
prediction_window = config['data']['prediction_window']
pred_df = pred_df.tail(prediction_window)
output_path = f'outputs/{model_name}/prediction_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
import os
os.makedirs(f'outputs/{model_name}', exist_ok=True)
pred_df.to_csv(output_path, index=False)

print(f"预测结果已保存到 {output_path}")