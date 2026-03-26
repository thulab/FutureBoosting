"""
Moirai2 模型适配器
"""
import warnings
import torch
import numpy as np
from typing import Union, Optional, Dict, Any
from ..base import BaseTimeSeriesModel


class Moirai2Adapter(BaseTimeSeriesModel):
    """
    Moirai2 模型适配器
    
    支持的模型路径示例：
    - Salesforce/moirai-2.0-R-small
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, model_type: str = 'moirai2'):
        """
        初始化 Moirai2 适配器
        
        Args:
            model_path: Huggingface 模型ID或本地路径，默认为 'salesforce/moirai-2.0-base'
            device: 设备
        """
        self.model_type = model_type
        if model_path is None:
            model_path = 'Salesforce/moirai-2.0-R-small'
        super().__init__('moirai2', model_path, device)
    
    def load_model(self):
        """加载 Moirai2 模型"""
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            if self.model_path is None:
                raise ValueError("Moirai2 model requires model_path to be specified")

            # self.model = Moirai2Module.from_pretrained(
            #         f"Salesforce/moirai-2.0-R-small",
            # ),
            self.module = Moirai2Module.from_pretrained(
                self.model_path,
            )
            self.forecast = Moirai2Forecast
            self._is_loaded = True
            print(f"Loaded Moirai2 model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Moirai2 model: {e}")
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Moirai2 进行预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度
            num_samples: 采样次数
            **kwargs: 其他参数
        
        Returns:
            dict: 预测结果
        """
        if num_samples != 9:
            warnings.warn(f"Moirai2 is restricted to sample 9 times as its default setting (relevant to module output dimensions), setting num_samples to 9")
            num_samples = 9
        self._ensure_model_loaded()
        B, C, L = context.shape
        context = self._normalize_input_dim_3D(context)
        with torch.no_grad():
            # Moirai2 的输入格式可能需要调整
            self.model = self.forecast(
                module=self.module,
                prediction_length=forecast_horizon,
                context_length=L,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).to(self.device)

            outputs = []
            for i in range(context.shape[1]):
                output = self.model.predict(context[:, i, :].cpu().numpy())
                
                # output = np.mean(output, axis=1)
                outputs.append(torch.Tensor(output).to(context.device))

        
        # forecast_tensor = torch.stack(outputs, dim=-1).reshape(B, C, forecast_horizon, num_samples) # [batch_size, quantile_num,forecast_horizon, n_variates]
        forecast_tensor = torch.stack(outputs, dim=-1).permute(0,3,2,1) # [batch_size, forecast_horizon, n_variates, num_samples]
        
        # 计算均值
        mean_forecast = forecast_tensor.mean(dim=-1)  # [batch_size, forecast_horizon, n_variates]

        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': None,
            'metadata': {
                'model': 'moirai2',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon
            }
        }
