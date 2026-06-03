from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        """预测，返回的DataFrame必须包含'ds'和'yhat'列"""
        pass

    def save(self, path: str):
        """保存模型到文件，子类按需实现"""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        """从文件加载模型，子类按需实现"""
        raise NotImplementedError