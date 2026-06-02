import yaml
import pandas as pd
from src.models.model_registry import get_model

# 1. 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
model_params = config['model']['params']

# 2. 加载模型
model = get_model(model_name, **model_params).load('models/latest.pkl')

# 3. 构造未来特征（例如预测未来24小时）
future_horizon = model_params.get('horizon', 24)
last_time = pd.Timestamp('2026-01-09 07:00:00')  # 模拟最后时间点
future_index = pd.date_range(start=last_time + pd.Timedelta(hours=1),
                             periods=future_horizon, freq='h')
X_future = pd.DataFrame(index=future_index)
X_future['hour'] = future_index.hour
# 注意：真实环境需要用最新的滞后值填充，这里简单填0作为演示
X_future['lag_1'] = 0
X_future['lag_24'] = 0

# 4. 预测
pred_df = model.predict(X_future)
output_path = f'outputs/prediction_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
import os
os.makedirs('outputs', exist_ok=True)
pred_df.to_csv(output_path, index=False)
print(f"预测结果已保存到 {output_path}")