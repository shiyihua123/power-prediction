import pandas as pd
import numpy as np
from .base import BaseModel
from .model_registry import register_model

@register_model("Demo")
class DemoModel(BaseModel):
    """持久性模型：用最后的观测值重复作为预测"""
    def __init__(self, config):
        pass

    def fit(self, df_full):
        pass

    def predict(self, futr_df):
        pass

    def cross_validate(self, df_full):
        pass

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str, config=None):
        pass