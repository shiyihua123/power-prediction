import yaml
from src.data_pipeline import prepare_data, build_future_df

with open('config.yaml') as f:
    config = yaml.safe_load(f)

df_full = prepare_data(config)

# 使用封装好的函数构建未来 df
future_df = build_future_df(df_full, config)

