import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, HuberLoss
from .base import BaseModel
from .model_registry import register_model

# 启用 Tensor Cores 以获得更好的性能
torch.set_float32_matmul_precision('medium')


@register_model("NHITS")
class NHITSModel(BaseModel):
    """封装 NeuralForecast 的 NHITS 模型"""
    def __init__(self, config):
        self.config = config
        self.freq = config['data']['freq']
        time_features = config['data']['feature_kwargs']['time_features']
        future_features = config['data']['feature_kwargs']['future_features']
        delta_features = config['data']['feature_kwargs']['delta_features']
        
        futr_exog_list = (time_features if time_features else [])  + (future_features if future_features else [])
        hist_exog_list = list(config['data']['files'].keys()) + (delta_features if delta_features else [])
        self.target_col = config['data']['target_col']
        hist_exog_list.remove(self.target_col)
        if config['training']['loss'] == 'mae':
            loss = MAE()
        elif config['training']['loss'] == 'huber':
            loss = HuberLoss(5)
        else:
            raise ValueError(f"Invalid loss function: {config['training']['loss']}")


        nhits = NHITS(
            h = config['horizon_total'],
            input_size=config['model']['params']['input_size'],
            max_steps=config['model']['params']['max_steps'],
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            loss=loss,
            valid_loss=loss,
            learning_rate=config['training']['learning_rate'],
            devices=[int(config['training']['devices'])],
            early_stop_patience_steps=config['model']['params']['early_stop_patience_steps'],
            scaler_type='robust',
            windows_batch_size=config['model']['params']['windows_batch_size'],
            inference_windows_batch_size=config['model']['params']['inference_windows_batch_size'],
            n_blocks=config['model']['params']['n_blocks'],
            n_pool_kernel_size=config['model']['params']['n_pool_kernel_size'],
            dropout_prob_theta=config['model']['params']['dropout_prob_theta'],
        )
        self.model = NeuralForecast(
            models=[nhits], 
            freq=self.freq
        )

        self.train_start = pd.to_datetime(self.config['data']['train']['start']).tz_localize('UTC')
        self.train_end = pd.to_datetime(self.config['data']['train']['end']).tz_localize('UTC')
        self.val_start = pd.to_datetime(self.config['data']['val']['start']).tz_localize('UTC')
        self.val_end = pd.to_datetime(self.config['data']['val']['end']).tz_localize('UTC')
        self.test_start = pd.to_datetime(self.config['data']['test']['start']).tz_localize('UTC')
        self.test_end = pd.to_datetime(self.config['data']['test']['end']).tz_localize('UTC')

        # self.train_size = len(pd.date_range(start=self.train_start, end=self.train_end, freq=self.freq))
        # self.val_size = len(pd.date_range(start=self.val_start, end=self.val_end, freq=self.freq))
        self.test_size = len(pd.date_range(start=self.test_start, end=self.test_end, freq=self.freq))

    def fit(self, df_full):
        df = self._to_long_format(df_full)
        df = df[df['ds'] >= self.train_start]
        
        self.model.fit(df=df, val_size=self.config['horizon_total'])
        return self

    def predict(self, futr_df):
        df = self._to_long_format(futr_df)
        fcst = self.model.predict(futr_df=df)
        return fcst

    
    def select_daily_cv_windows(
        self,
        cv_results: pd.DataFrame,
        prediction_window: int,
        local_tz: str,
        start_hour: int = 0,
    ):
        cv_results = cv_results.copy()
        cv_results["ds_local"] = cv_results["ds"].dt.tz_convert(local_tz)

        # 筛选指定小时开始的 cutoff
        valid_cutoffs = (
            cv_results
            .groupby("cutoff")["ds_local"]
            .min()
            .loc[lambda s: s.dt.hour == start_hour]
            .index
        )

        cv_selected = (
            cv_results[cv_results["cutoff"].isin(valid_cutoffs)]
            .groupby("cutoff")
            .tail(prediction_window)
        )

        cv_selected["begin_utc"] = cv_selected.groupby("cutoff")["ds"].transform("first")
        cv_selected = cv_selected.drop(columns=["ds_local"])

        return cv_selected
    

    def cross_validate(self, df_full):
        df = self._to_long_format(df_full)
        df = df[df['ds'] >= self.train_start]
        df = df[df['ds'] <= self.test_end]
        # # 找出有nan的行
        # nan_rows = df[df.isnull().any(axis=1)]

        cv_results = self.model.cross_validation(
            df=df,
            n_windows=None,
            val_size=self.config['horizon_total'],
            test_size=self.test_size
        )

        # 转成本地时区
        local_tz = self.config['data']['feature_kwargs']['local_tz']
        prediction_window = self.config['data']['prediction_window']
        cv_results['ds'] = cv_results['ds'].dt.tz_convert(local_tz)

        cv_selected = self.select_daily_cv_windows(
            cv_results=cv_results,
            prediction_window=prediction_window,
            local_tz=local_tz,
            start_hour=int(self.config['data']['insured_time']),
        )
        cv_selected['ds'] = cv_selected['ds'].dt.tz_convert("UTC")
        cv_selected['begin_utc'] = cv_selected['begin_utc'].dt.tz_convert("UTC")
        # 删除cutoff列
        cv_selected = cv_selected.drop(columns=['cutoff', 'unique_id'])

        return cv_selected

    def _to_long_format(self, df_full):
        """将宽表 df_full 转换为 NeuralForecast 需要的长格式"""
        df = df_full.copy()

        df['unique_id'] = 'series_1'
        # 将时间列重命名为 'ds'
        time_col = self.config['data']['raw_col'][0]
        df = df.rename(columns={time_col: 'ds'})
        if self.target_col in df.columns:
            df = df.rename(columns={self.target_col: 'y'})

        return df

    def save(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str, config=None):
        """加载保存的模型（走完整初始化流程）
        
        Args:
            path: 模型文件路径
            config: 配置字典（必需，用于初始化模型结构）
        
        Returns:
            加载了预训练模型的 NHITSModel 对象
        """
        if config is None:
            raise ValueError("加载模型时必须提供 config 参数")
        
        import joblib
        
        # 1. 走完整的初始化流程
        obj = cls(config)
        
        # 2. 用加载的预训练模型替换初始化时创建的模型
        loaded_model = joblib.load(path)
        obj.model = loaded_model
        
        return obj
