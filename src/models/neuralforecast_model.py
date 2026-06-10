import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast import models as nf_models
from neuralforecast.losses.pytorch import MAE, HuberLoss
from .base import BaseModel
from .model_registry import register_model
from typing import List, Union


# 固定 GPU 计算的确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 启用 Tensor Cores 以获得更好的性能
torch.set_float32_matmul_precision('medium')


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
        self.model_name = config['model_name']
        try:
            model_cls = getattr(nf_models, self.model_name)
        except AttributeError:
            available = [m for m in dir(nf_models) if not m.startswith('_')]
            raise ValueError(f"未知的 NeuralForecast 模型: '{self.model_name}'，可用: {available}")

        # 特征列表
        time_features = config['data']['feature_kwargs'].get('time_features', []) or []
        future_features = config['data']['feature_kwargs'].get('future_features', []) or []
        delta_features = config['data']['feature_kwargs'].get('delta_features', []) or []



        futr_exog_list = time_features + future_features
        self.future_features = future_features
        hist_exog_list = list(config['data']['files'].keys()) + delta_features
        self.target_col = config['data']['target_col']
        hist_exog_list.remove(self.target_col)
        for feature in future_features:
            hist_exog_list.remove(feature)

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
        model_params = dict(model_config[self.model_name]['params'])
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
        fcst = fcst.drop(columns=['unique_id'])
        fcst.rename(columns={'ds': 'date', self.model_name: 'ours'}, inplace=True)
        return fcst


    def extract_sequences_from_midnight(
        self,
        df: pd.DataFrame,
        length: int,
        time_col: str = 'ds_local',
        start_hour: int = 0,
        return_as_list_of_dfs: bool = True
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        根据本地时间列（如 ds_local）选出每日指定小时（默认0点）作为起点，
        往后取固定长度的连续时间序列样本，仅保留完整长度（即起点后 length-1 个点都存在且时间连续）。

        参数:
            df: 包含时间列的 DataFrame，假设已按时间排序（未排序时会自动排序）
            length: 序列长度（包含起点）
            time_col: 时间列名称，需为带时区的本地时间（如 '2025-01-01 00:00:00+01:00'）
            start_hour: 起始小时（24小时制，默认为0）
            return_as_list_of_dfs: 若 True，返回 DataFrame 列表；若 False，返回单个 DataFrame 并增加 'sample_id' 列

        返回:
            若 return_as_list_of_dfs=True，返回 List[pd.DataFrame]，每个元素是一个长度为 length 的序列样本
            否则返回 pd.DataFrame，包含所有有效样本，并带有 'sample_id' 列标识不同的序列

        示例:
            samples = extract_sequences_from_midnight(df, length=24)
            # 或者
            df_samples = extract_sequences_from_midnight(df, length=24, return_as_list_of_dfs=False)
        """
        # 复制并确保按时间升序排序
        df = df.copy().sort_values(time_col).reset_index(drop=True)
        
        # 提取本地时间的日期和小时
        dt_local = pd.to_datetime(df[time_col])
        df['_date'] = dt_local.dt.date
        df['_hour'] = dt_local.dt.hour
        
        # 找出所有满足起始小时的行索引
        start_mask = df['_hour'] == start_hour
        start_indices = df.index[start_mask].tolist()
        
        valid_sequences = []
        sample_id = 0
        
        for start_idx in start_indices:
            # 检查从 start_idx 开始往后是否足够 length 条记录
            end_idx = start_idx + length - 1
            if end_idx >= len(df):
                continue  # 长度不足，跳过
            
            # 提取候选序列
            candidate = df.iloc[start_idx: end_idx + 1].copy()
            
            # 检查时间连续性：相邻两条记录的时间差应为1小时（考虑夏令时，使用本地时间差）
            time_vals = pd.to_datetime(candidate[time_col])
            time_diffs = time_vals.diff().iloc[1:]  # 跳过第一个NaN
            # 由于夏令时可能存在 1小时或2小时（切换时）? 实际上本地时间间隔始终是1小时，但时区偏移变化会导致UTC间隔变化
            # 但这里比较的是本地时间差，应该总是1小时（除非数据缺失）。我们使用 pd.Timedelta(hours=1) 判断
            if not (time_diffs == pd.Timedelta(hours=1)).all():
                continue  # 时间不连续，丢弃该样本
            
            # 可选：验证起点确实是 start_hour（已经通过筛选保证），且起点日期没有因为时区变化出现重复？
            # 如果起点存在重复（例如夏令时回退时的2:30？但这里是整点0点，通常不会重复），可忽略
            
            # 移除辅助列
            candidate = candidate.drop(columns=['_date', '_hour'])
            
            if return_as_list_of_dfs:
                valid_sequences.append(candidate)
            else:
                candidate['sample_id'] = sample_id
                valid_sequences.append(candidate)
                sample_id += 1
        
        if return_as_list_of_dfs:
            return valid_sequences
        else:
            if valid_sequences:
                return pd.concat(valid_sequences, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns.tolist() + ['sample_id'])

    
    def conduct_cross_test_windows(self, df, local_tz):
        df["ds_local"] = df["ds"].dt.tz_convert(local_tz)
        
        test_samples = self.extract_sequences_from_midnight(
            df=df,
            length=self.config['horizon_total'],
            time_col='ds_local',
            start_hour=int(self.config['data']['insured_time']),
            return_as_list_of_dfs=True,
        )
        prediction_window = self.config['data']['prediction_window']
        # 填充未来特征
        final_samples = []
        for sample in test_samples:
            for feature in self.future_features:
                feature_forecast_file_path = self.config['data']['forecast_file'][feature]
                print(feature_forecast_file_path)
                print(sample)
                sample.drop(columns=['y', 'ds_local'], inplace=True)
            final_samples.append(sample)
            
            break
        return final_samples
        pass
    def select_daily_cv_windows(
        self,
        cv_results: pd.DataFrame,
        prediction_window: int,
        local_tz: str,
        start_hour: int = 0,
    ):
        cv_results = cv_results.copy()
        cv_results["ds_local"] = cv_results["ds"].dt.tz_convert(local_tz)

        min_ds_by_cutoff = cv_results.groupby("cutoff")["ds_local"].min()

        valid_cutoffs = min_ds_by_cutoff.loc[min_ds_by_cutoff.dt.hour == start_hour].index
        
        cv_selected = cv_results[cv_results["cutoff"].isin(valid_cutoffs)]

        cv_selected = cv_selected.drop(columns=["ds_local"])

        cv_selected['ds'] = cv_selected['ds'].dt.tz_convert("UTC")

        cv_selected = cv_selected.groupby("cutoff").tail(prediction_window)

        cv_selected["begin_utc"] = cv_selected.groupby("cutoff")["ds"].transform("first")

        cv_selected = cv_selected.drop(columns=['cutoff', 'unique_id'])

        return cv_selected

    def cross_validate1(self, df_full):
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
        
        local_tz = self.config['data']['feature_kwargs']['local_tz']
        prediction_window = self.config['data']['prediction_window']
        cv_results['ds'] = cv_results['ds'].dt.tz_convert(local_tz)

        cv_selected = self.select_daily_cv_windows(
            cv_results=cv_results,
            prediction_window=prediction_window,
            local_tz=local_tz,
            start_hour=int(self.config['data']['insured_time']),
        )

        return cv_selected


    def cross_validate(self, df_full):
        df = self._to_long_format(df_full)
        
        df_train = df[(df['ds'] >= self.train_start) & (df['ds'] <= self.val_end)]
        df_test = df[(df['ds'] >= self.test_start) & (df['ds'] <= self.test_end)]

        val_size = self.config['horizon_total']

        print("训练数据范围:", df_train['ds'].min(), "~",df_train['ds'].max())
        print(f"验证数据为训练集后{val_size}个时间点")
        print("测试数据范围:", df_test['ds'].min(), "~",df_test['ds'].max())
        
        self.model.fit(df=df_train, val_size=val_size)
        # print(df_test)
        local_tz = self.config['data']['feature_kwargs']['local_tz']
        prediction_window = self.config['data']['prediction_window']
        final_samples = self.conduct_cross_test_windows(
            df_test,
            local_tz=local_tz,
        )
        for sample in final_samples:
            yhat = self.model.predict(futr_df=sample)
            print(yhat)
        exit(0)
        # 构造预测样本
        
        # local_tz = self.config['data']['feature_kwargs']['local_tz']
        # prediction_window = self.config['data']['prediction_window']
        # cv_results['ds'] = cv_results['ds'].dt.tz_convert(local_tz)

        # cv_selected = self.select_daily_cv_windows(
        #     cv_results=cv_results,
        #     prediction_window=prediction_window,
        #     local_tz=local_tz,
        #     start_hour=int(self.config['data']['insured_time']),
        # )

        # return cv_selected

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
