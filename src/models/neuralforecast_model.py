import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast import models as nf_models
from neuralforecast.losses.pytorch import MAE, HuberLoss
from .base import BaseModel
from .model_registry import register_model

# 固定 GPU 计算的确定性（cuDNN 默认会选择非确定性算法以优化速度）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 启用 Tensor Cores 以获得更好的性能
torch.set_float32_matmul_precision('medium')


@register_model("BiTCN")
@register_model("CNN")
@register_model("DeepNPTS")
@register_model("DilatedRNN")
@register_model("GRU")
@register_model("HINT")
@register_model("KAN")
@register_model("LSTM")
@register_model("MLP")
@register_model("MLPMultivariate")
@register_model("NBEATSx")
@register_model("NHITS")
@register_model("RNN")
@register_model("TCN")
@register_model("TFT")
@register_model("TiDE")
@register_model("TSMixerx")
@register_model("VanillaTransformer")
@register_model("XLinear")
@register_model("xLSTM")
class NeuralForecastModel(BaseModel):
    """通用 NeuralForecast 模型封装，支持所有 neuralforecast.models 中的模型"""

    def __init__(self, config, model_config):
        self.model_config = model_config
        
        self.config = config
        self.freq = config['data']['freq']
        
        # 获取目标 NF 模型类
        model_name = config['model_name']
        try:
            model_cls = getattr(nf_models, model_name)
        except AttributeError:
            available = [m for m in dir(nf_models) if not m.startswith('_')]
            raise ValueError(f"未知的 NeuralForecast 模型: '{model_name}'，可用: {available}")

        # 特征列表
        time_features = config['data']['feature_kwargs'].get('time_features', []) or []
        future_features = config['data']['feature_kwargs'].get('future_features', []) or []
        delta_features = config['data']['feature_kwargs'].get('delta_features', []) or []



        futr_exog_list = time_features + future_features
        hist_exog_list = list(config['data']['files'].keys()) + delta_features
        self.target_col = config['data']['target_col']
        hist_exog_list.remove(self.target_col)

        print(f"futr_exog_list: {futr_exog_list}")
        print(f"hist_exog_list: {hist_exog_list}")

        # 损失函数
        loss_name = config['training']['loss']
        if loss_name == 'mae':
            loss = MAE()
        elif loss_name == 'huber':
            loss = HuberLoss(5)
        else:
            raise ValueError(f"未知的损失函数: '{loss_name}'，可用: mae, huber")
        
        # 构建模型参数：先从 config params 取，再覆盖通用参数
        model_params = dict(model_config[model_name]['params'])
        model_params.update({
            'h': config['horizon_total'],
            'futr_exog_list': futr_exog_list,
            'hist_exog_list': hist_exog_list,
            'loss': loss,
            'valid_loss': loss,
            'learning_rate': config['training']['learning_rate'],
            'scaler_type': config['training']['scaler_type'],
            'random_seed': config['training']['seed'],
        })

        # devices 特殊处理：转为列表
        devices = config['training'].get('devices')
        if devices is not None and str(devices).lower() != 'cpu':
            model_params['devices'] = [int(devices)]

        nf_model = model_cls(**model_params)
        self.model = NeuralForecast(models=[nf_model], freq=self.freq)

        # 时间范围
        self.train_start = pd.to_datetime(self.config['data']['train']['start']).tz_localize('UTC')
        self.train_end = pd.to_datetime(self.config['data']['train']['end']).tz_localize('UTC')
        self.val_start = pd.to_datetime(self.config['data']['val']['start']).tz_localize('UTC')
        self.val_end = pd.to_datetime(self.config['data']['val']['end']).tz_localize('UTC')
        self.test_start = pd.to_datetime(self.config['data']['test']['start']).tz_localize('UTC')
        self.test_end = pd.to_datetime(self.config['data']['test']['end']).tz_localize('UTC')

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
        print("训练数据范围:", df['ds'].min(), "~",df['ds'].max())
        print(f"验证数据为训练集后{self.config['horizon_total']}个时间点")
        cv_results = self.model.cross_validation(
            df=df,
            n_windows=None,
            val_size=self.config['horizon_total'],
            test_size=self.test_size
        )
        print(cv_results)
        
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
        cv_selected = cv_selected.drop(columns=['cutoff', 'unique_id'])

        return cv_selected

    def _to_long_format(self, df_full):
        """将宽表 df_full 转换为 NeuralForecast 需要的长格式"""
        df = df_full.copy()
        df['unique_id'] = 'series_1'
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
    def load(cls, path: str, config=None, model_config=None):
        """加载保存的模型"""
        if config is None:
            raise ValueError("加载模型时必须提供 config 参数")

        import joblib
        obj = cls(config, model_config)
        loaded_model = joblib.load(path)
        obj.model = loaded_model
        return obj
