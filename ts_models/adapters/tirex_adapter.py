"""
TiRex 模型适配器
"""
import warnings
import torch
import numpy as np
from typing import Union, Optional, Dict, Any
from ..base import BaseTimeSeriesModel


class TiRexAdapter(BaseTimeSeriesModel):
    """
    TiRex 模型适配器
    
    支持的模型路径示例：
    - NX-AI/TiRex
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, model_type: str = 'tirex'):
        """
        初始化 TiRex 适配器
        
        Args:
            model_path: Huggingface 模型ID或本地路径
            device: 设备
        """
        self.model_type = model_type
        if model_path is None:
            model_path = 'NX-AI/TiRex'
        super().__init__('tirex', model_path, device)
    
    def load_model(self):
        """加载 TiRex 模型"""
        try:
            from tirex import load_model, ForecastModel

            if self.model_path is None:
                raise ValueError("TiRex model requires model_path to be specified")

            self.model : ForecastModel = load_model(
                'NX-AI/TiRex', 
                device=self.device,
                hf_kwargs={
                    'cache_dir': self.model_path,
                }
            )
            self._is_loaded = True
            print(f"Loaded TiRex ({self.model_type}) model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TiRex ({self.model_type}) model: {e}")
    
    def predict(
        self, 
        context: Union[torch.Tensor, np.ndarray], 
        forecast_horizon: int, 
        num_samples: Optional[int] = 1, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 TiRex 进行预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度
            num_samples: 采样次数
            **kwargs: 其他参数
        
        Returns:
            dict: 预测结果
        """
        if num_samples != 9:
            warnings.warn(f"TiRex is restricted to sample 9 times as its default setting (relevant to module output dimensions), setting num_samples to 9")
            num_samples = 9
        self._ensure_model_loaded()
        
        B, C, L = context.shape
        context = self._normalize_input_dim_2D(context)
        
        with torch.no_grad():
            quantiles, mean = self.model.forecast(
                context = context,
                prediction_length = forecast_horizon
            )

        
        forecast_tensor = quantiles.reshape(B, C, forecast_horizon, num_samples) # [batch_size, forecast_horizon, quantile_num]
        mean_forecast = mean.reshape(B, C, forecast_horizon) # [batch_size, forecast_horizon]
        
        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': quantiles,
            'metadata': {
                'model': f'tirex-{self.model_type}',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon
            }
        }
