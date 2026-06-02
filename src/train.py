import yaml
from src.data_pipeline import load_demo_data
from src.models.model_registry import get_model

# 1. 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
model_params = config['model']['params']

# 2. 获取数据
X, y = load_demo_data()

# 3. 创建模型并训练
model = get_model(model_name, **model_params)
model.fit(X, y)

# 4. 保存模型
model.save('models/latest.pkl')
print(f"模型 {model_name} 训练完成，已保存到 models/latest.pkl")