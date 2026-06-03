import pandas as pd
import numpy as np
from .base import BaseModel
from .model_registry import register_model

@register_model("Naive")
class NaiveModel(BaseModel):
    """持久性模型：用最后的观测值重复作为预测"""
    def __init__(self, horizon=24):
        self.horizon = horizon
        self.last_values = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        # 只记住 y 的最后 horizon 个值
        self.last_values = y.values[-self.horizon:]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # X 的索引是未来时间戳
        n_future = len(X)
        # 循环填充 self.last_values 直到长度足够
        repeated = np.tile(self.last_values, int(np.ceil(n_future / len(self.last_values))))
        yhat = repeated[:n_future]
        return pd.DataFrame({'ds': X.index, 'yhat': yhat})

    def save(self, path: str):
        import joblib
        joblib.dump({'horizon': self.horizon, 'last_values': self.last_values}, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        data = joblib.load(path)
        obj = cls(horizon=data['horizon'])
        obj.last_values = data['last_values']
        return obj