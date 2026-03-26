"""
TabPFN 模型适配器
"""
import warnings
import torch
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime, timedelta
from ..base import BaseTimeSeriesModel



class TabPFNAdapter(BaseTimeSeriesModel):
    """
    TabPFN 模型适配器
    
    注意：TabPFN 主要用于表格数据，但也可以用于时序预测
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 TabPFN 适配器
        
        Args:
            model_path: 模型路径（TabPFN 通常不需要指定路径）
            device: 设备
        """
        super().__init__('tabpfn', model_path, device)
    
    def load_model(self):
        """加载 TabPFN 模型"""
        try:
            from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

            self.model = TabPFNTSPipeline(
                tabpfn_mode=TabPFNMode.LOCAL,  # adapt this to TabPFNMode.CLIENT if using API
            )
            self._is_loaded = True
            print("Loaded TabPFN model")
        except ImportError as e:
            raise ImportError(
                "TabPFN Time Series requires the 'tabpfn-time-series' package. "
                "Install it with: pip install tabpfn-time-series"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load TabPFN model: {e}")
    
    def _array_to_timeseries_dataframe(
        self,
        context: np.ndarray,
        forecast_horizon: int,
        item_ids: Optional[List[str]] = None,
        start_timestamp: Optional[datetime] = None,
        freq: str = 'D'
    ):
        """
        将 numpy 数组转换为 TimeSeriesDataFrame
        
        Args:
            context: 输入数组 [batch_size, seq_len] 或 [seq_len]
            item_ids: 每个序列的ID列表，如果为None则自动生成
            start_timestamp: 起始时间戳，如果为None则使用当前时间
            freq: 时间频率（'D'=天, 'H'=小时, 'T'=分钟等）
        
        Returns:
            TimeSeriesDataFrame: 转换后的时序数据框
        """
        from tabpfn_time_series import TimeSeriesDataFrame
        
        # 确保是2D数组
        if context.ndim == 1:
            context = context.reshape(1, -1)
        
        batch_size, seq_len = context.shape
        
        # 生成 item_ids
        if item_ids is None:
            item_ids = [f"item_{i}" for i in range(batch_size)]
        elif len(item_ids) != batch_size:
            raise ValueError(f"item_ids length ({len(item_ids)}) must match batch_size ({batch_size})")
        
        # 生成时间戳
        if start_timestamp is None:
            start_timestamp = datetime(2024, 1, 1)  # 使用确定的时间戳
        
        # 创建 pandas DataFrame
        data_list = []
        future_list = []
        for i, item_id in enumerate(item_ids):
            timestamps = pd.date_range(
                start=start_timestamp,
                periods=seq_len,
                freq=freq
            )

            # 从最后一个时间戳的下一个时间点开始生成预测时间戳
            # 使用 pd.tseries.frequencies.to_offset 将频率字符串转换为 DateOffset
            from pandas.tseries.frequencies import to_offset
            freq_offset = to_offset(freq)
            forecast_timestamps = pd.date_range(
                start=timestamps[-1] + freq_offset,  # 从最后一个时间戳的下一个时间点开始
                periods=forecast_horizon,
                freq=freq
            )
            for t, value in zip(timestamps, context[i]):
                data_list.append({
                    'item_id': item_id,
                    'timestamp': t,
                    'target': float(value)
                })
            for t in forecast_timestamps:
                future_list.append({
                    'item_id': item_id,
                    'timestamp': t,
                })
        
        df = pd.DataFrame(data_list)
        future_df = pd.DataFrame(future_list)
        # 转换为 TimeSeriesDataFrame
        # TimeSeriesDataFrame 通常需要 MultiIndex (item_id, timestamp)
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column='item_id',
            timestamp_column='timestamp'
        )
        
        return ts_df, future_df
    
    def _timeseries_dataframe_to_dataframe(
        self,
        ts_df: Union['TimeSeriesDataFrame', pd.DataFrame]
    ) -> pd.DataFrame:
        """
        将 TimeSeriesDataFrame 转换为标准的 pandas DataFrame
        
        Args:
            ts_df: TimeSeriesDataFrame 对象
        
        Returns:
            pd.DataFrame: 标准的 pandas DataFrame，包含 item_id, timestamp, target 列
        """
        # TimeSeriesDataFrame 是 pandas DataFrame 的子类，通常有 MultiIndex
        # 重置索引以获取 item_id 和 timestamp 作为列
        df = ts_df.reset_index()
        
        # 确保列名正确
        if 'item_id' not in df.columns and 'item_id' in ts_df.index.names:
            # 如果 item_id 在索引中，应该已经在 reset_index 后出现
            pass
        
        return df
    
    def _timeseries_dataframe_to_array(
        self,
        ts_df: Union['TimeSeriesDataFrame', pd.DataFrame],
        item_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        将 TimeSeriesDataFrame 转换为 numpy 数组
        
        Args:
            ts_df: TimeSeriesDataFrame 对象
            item_ids: 要提取的 item_id 列表，如果为None则提取所有
        
        Returns:
            np.ndarray: 数组 [batch_size, seq_len] 或 [seq_len]
        """
        df = self._timeseries_dataframe_to_dataframe(ts_df)
        
        if item_ids is None:
            item_ids = df['item_id'].unique()
        
        arrays = []
        for item_id in item_ids:
            item_data = df[df['item_id'] == item_id]['target'].values
            arrays.append(item_data)
        
        result = np.array(arrays)
        
        # 如果只有一个序列，返回1D数组
        if len(arrays) == 1:
            return result[0]
        
        return result
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        item_ids: Optional[List[str]] = None,
        start_timestamp: Optional[datetime] = None,
        freq: str = 'D',
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 TabPFN 进行零样本时序预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度（未来时间步数）
            num_samples: 采样次数（TabPFN 可能支持多样本预测）
            item_ids: 每个序列的ID列表，如果为None则自动生成
            start_timestamp: 起始时间戳，如果为None则使用当前时间
            freq: 时间频率（'D'=天, 'H'=小时, 'T'=分钟等）
            **kwargs: 其他参数，传递给 TabPFNTimeSeriesPredictor.predict
        
        Returns:
            dict: 预测结果，包含：
                - 'forecast': torch.Tensor, shape [batch_size, num_samples, forecast_horizon]
                - 'mean': torch.Tensor, shape [batch_size, forecast_horizon]
                - 'quantiles': None (TabPFN 可能不直接提供分位数)
                - 'metadata': dict - 元数据信息
        """
        if num_samples != 9:
            warnings.warn(f"TabPFN is restricted to sample 9 times as its default setting (relevant to module output dimensions), setting num_samples to 9")
            num_samples = 9
        self._ensure_model_loaded()
        
        B, C, L = context.shape

        # 标准化输入格式
        context = self._normalize_input_dim_2D(context)
        batch_size = context.shape[0]
        
        # 转换为 numpy 数组
        context_np = context.cpu().numpy()
        
        # 生成 item_ids
        if item_ids is None:
            item_ids = [f"item_{i}" for i in range(batch_size)]
        
        # 转换为 TimeSeriesDataFrame
        ts_df, future_df = self._array_to_timeseries_dataframe(
            context_np,
            forecast_horizon=forecast_horizon,
            item_ids=item_ids,
            start_timestamp=start_timestamp,
            freq=freq
        )

        
        # 使用 TabPFN 进行预测
        try:
            # TabPFNTimeSeriesPredictor 的 predict 方法
            # for i, item_id in enumerate(item_ids):

                # item_id_level = ts_df.index.names.index('item_id')
                # context_df = ts_df.xs(item_id, level=item_id_level, drop_level=False)
                # # context_df = ts_df.loc[item_id]
            predictions_ts_df = self.model.predict_df(
                context_df = ts_df,
                future_df=future_df,
                # prediction_length=forecast_horizon,
            )
        except Exception as e:
            raise RuntimeError(f"TabPFN prediction failed: {e}") from e
        
        # 将 predictions_ts_df 转换为所需的格式
        # predictions_ts_df 格式: MultiIndex (item_id, timestamp), 列包括 target, 0.1, 0.2, ..., 0.9
        
        # 提取所有 item_ids（按原始顺序）
        if isinstance(predictions_ts_df.index, pd.MultiIndex):
            unique_item_ids = predictions_ts_df.index.get_level_values('item_id').unique()
            # 保持原始 item_ids 的顺序
            item_ids_in_pred = [item_id for item_id in item_ids if item_id in unique_item_ids]
        else:
            item_ids_in_pred = item_ids
        
        # 提取分位数列（0.1 到 0.9）
        # 列名可能是数字（float/int）或字符串格式的数字
        quantile_columns = []
        for col in predictions_ts_df.columns:
            if col == 'target':
                continue  # 跳过 target 列
            try:
                # 尝试将列名转换为浮点数
                float_val = float(col)
                if 0.0 <= float_val <= 1.0:  # 分位数应该在 0-1 之间
                    quantile_columns.append(col)
            except (ValueError, TypeError):
                pass
        
        # 按分位数值排序
        quantile_columns = sorted(quantile_columns, key=lambda x: float(x))
        
        # 如果找不到分位数列，尝试使用 'target' 列
        if not quantile_columns:
            quantile_columns = ['target']
        
        all_forecasts = []
        all_means = []
        
        for item_id in item_ids_in_pred:
            # 提取当前 item_id 的所有预测数据
            if isinstance(predictions_ts_df.index, pd.MultiIndex):
                item_data = predictions_ts_df.xs(item_id, level='item_id', drop_level=False)
                # 按时间戳排序（xs 后可能只剩下 timestamp level，或仍保持 MultiIndex）
                if isinstance(item_data.index, pd.MultiIndex) and 'timestamp' in item_data.index.names:
                    item_data = item_data.sort_index(level='timestamp')
                else:
                    item_data = item_data.sort_index()
            else:
                item_data = predictions_ts_df[predictions_ts_df.index.get_level_values('item_id') == item_id]
                item_data = item_data.sort_index()
            
            # 提取分位数数据作为 samples
            # shape: [forecast_horizon, num_quantiles]
            quantile_values = item_data[quantile_columns].values
            
            # 转置为 [num_quantiles, forecast_horizon]
            quantile_values = quantile_values.T
            
            # 转换为 torch.Tensor
            quantile_tensor = torch.from_numpy(quantile_values).float()  # [num_quantiles, forecast_horizon]
            
            all_forecasts.append(quantile_tensor)
            
            # 提取 mean（使用 target 列，如果不存在则使用 0.5 分位数或所有分位数的均值）
            if 'target' in item_data.columns:
                mean_values = item_data['target'].values
            else:
                # 尝试找到 0.5 分位数（可能是 0.5 或 '0.5'）
                median_col = None
                for col in quantile_columns:
                    try:
                        if abs(float(col) - 0.5) < 1e-6:  # 检查是否是 0.5
                            median_col = col
                            break
                    except (ValueError, TypeError):
                        continue
                
                if median_col is not None:
                    mean_values = item_data[median_col].values
                else:
                    # 如果没有 target 或 0.5，使用所有分位数的均值
                    mean_values = quantile_values.mean(axis=0)
            
            mean_tensor = torch.from_numpy(mean_values).float()  # [forecast_horizon]
            all_means.append(mean_tensor)
        
        # 堆叠所有batch的预测结果
        # all_forecasts: List of [num_quantiles, forecast_horizon]
        forecast_tensor = torch.stack(all_forecasts, dim=0).reshape(B, C, forecast_horizon, num_samples)  # [batch_size, num_quantiles, forecast_horizon]
        
        # 堆叠所有batch的均值
        mean_forecast = torch.stack(all_means, dim=0).reshape(B, C, forecast_horizon)  # [batch_size, forecast_horizon]
        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': None,  # TabPFN 可能不直接提供分位数预测
            'metadata': {
                'model': 'tabpfn',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon,
                'item_ids': item_ids,
                'freq': freq,
                'note': 'TabPFN零样本时序预测：使用TimeSeriesDataFrame格式'
            }
        }
