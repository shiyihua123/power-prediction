import pandas as pd
import numpy as np
import lightgbm as lgb
from .base import BaseModel
from .model_registry import register_model
from sklearn.metrics import mean_absolute_error

@register_model("LightGBM")
class LightGBMModel(BaseModel):
    """LightGBM 回归模型，只使用时间特征预测电价。"""

    def __init__(self, config):
        self.config = config
        self.model_name = config['model']['name']
        self.target_col = config['data']['target_col']
        self.time_col = config['data']['raw_col'][0]
        self.freq = config['data']['freq']

        # 特征列：只取配置里指定的时间特征
        time_features = config['data']['feature_kwargs'].get('time_features', []) or []
        self.feature_cols = list(time_features)

        # LightGBM 参数
        model_params = dict(config['model'].get('params', {}))
        self.lgb_params = model_params

        self.model = lgb.LGBMRegressor(**self.lgb_params)

        # 时间划分
        self.train_start = pd.to_datetime(config['data']['train']['start']).tz_localize('UTC')
        self.val_start = pd.to_datetime(config['data']['val']['start']).tz_localize('UTC')
        self.test_start = pd.to_datetime(config['data']['test']['start']).tz_localize('UTC')

    # ─────────── 训练 ───────────

    def fit(self, df_full: pd.DataFrame):
        #输出df的前五行
        # print(df.head())
        # pass
        df = df_full.sort_values(self.time_col).reset_index(drop=True)

        # 按时间划分
        train = df[df[self.time_col] < self.val_start]
        val = df[(df[self.time_col] >= self.val_start) & (df[self.time_col] < self.test_start)]

        X_train, y_train = train[self.feature_cols], train[self.target_col]
        X_val, y_val = val[self.feature_cols], val[self.target_col]

        print(f"特征列 ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"训练集: {len(X_train)} 条")
        print(f"验证集: {len(X_val)} 条")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # # 打印验证集结果
        y_pred = self.model.predict(X_val)
        print(f"验证集 MAE: {mean_absolute_error(y_val, y_pred):.4f}")
        return self

    # ─────────── 预测 ───────────

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        #打印future_df的全部数据
        print(future_df)
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        X = future_df[self.feature_cols]
        yhat = self.model.predict(X)
        return pd.DataFrame({"ds": future_df[self.time_col], self.model_name: yhat})

    # ─────────── 交叉验证 ───────────

    def cross_validate(self, df_full: pd.DataFrame) -> pd.DataFrame:
        df_full = df_full.sort_values(self.time_col).reset_index(drop=True)

        train = df_full[df_full[self.time_col] < self.test_start]
        test = df_full[(df_full[self.time_col] >= self.test_start)]

        # 重新训练
        val = df_full[(df_full[self.time_col] >= self.val_start) & (df_full[self.time_col] < self.test_start)]
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(
            train[self.feature_cols], train[self.target_col],
            eval_set=[(val[self.feature_cols], val[self.target_col])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)],
        )

        # 在测试集上预测
        y_pred = self.model.predict(test[self.feature_cols])

        cv_results = pd.DataFrame({
            "ds": test[self.time_col].values,
            "y": test[self.target_col].values,
            "LightGBM": y_pred,
        })

        # 按天切分窗口
        local_tz = self.config['data']['feature_kwargs']['local_tz']
        cv_results['ds'] = pd.to_datetime(cv_results['ds'])
        cv_selected = self._select_daily_windows(cv_results, local_tz)
        return cv_selected

    # ─────────── 保存 / 加载 ───────────

    def save(self, path: str):
        import os, joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str, config=None):
        """加载模型。

        Args:
            path: 模型文件路径
            config: 配置字典（必需，因为 __init__ 依赖它初始化参数）
        """
        if config is None:
            raise ValueError("加载模型时必须提供 config 参数")
        import joblib
        obj = cls(config)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.feature_cols = data["feature_cols"]
        return obj
