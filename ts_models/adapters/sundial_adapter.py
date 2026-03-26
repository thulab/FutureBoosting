"""
Sundial 模型适配器
"""
import torch
import numpy as np
from typing import Union, Optional, Dict, Any
from ..base import BaseTimeSeriesModel


class SundialAdapter(BaseTimeSeriesModel):
    """
    Sundial 模型适配器

    支持的模型路径示例：
    - sundial-base-128m
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, model_type: str = 'sundial'):
        """
        初始化 Sundial 适配器
        
        Args:
            model_path: 模型路径（本地路径或Huggingface路径）
            device: 设备
            model_type: 模型类型（'sundial', 'sundial_cora', 'sundial-dualweaver' 等）
        """
        self.model_type = model_type
        if model_path is None:
            model_path = 'thuml/sundial-base-128m' # [qiuyz] NOTE: this is the default model path, should be changed to the actual model path
        super().__init__('sundial', model_path, device)
    
    def load_model(self):
        """加载 Sundial 模型"""
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            
            if self.model_path is None:
                raise ValueError("Sundial model requires model_path to be specified")

            # [qiuyz] NOTE: Do we need loading configs from local dir or from hugginface?
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                trust_remote_code=True,
                device_map=self.device,
                dtype=torch.float32
            )
            self.model.eval()
            self._is_loaded = True
            print(f"Loaded Sundial ({self.model_type}) model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Sundial model: {e}")
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        revin: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Sundial 进行预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度
            num_samples: 采样次数
            revin: 是否使用reversible instance normalization
            **kwargs: 其他参数
        
        Returns:
            dict: 预测结果
        """
        self._ensure_model_loaded()
        B, C, L = context.shape
        
        context = self._normalize_input_dim_2D(context)
        
        with torch.no_grad():
            # 使用 Sundial 的 generate 方法
            if hasattr(self.model, 'generate'):
                outputs = self.model.generate(context, max_new_tokens=forecast_horizon, num_samples=num_samples, revin=revin)
                # forecast_tensor = torch.tensor(outputs)  # [batch_size, num_samples, forecast_horizon]
                forecast_tensor = outputs.detach().clone()

            else:
                # 使用 forward 方法（需要提供完整的输入）
                # 注意：这可能需要调整
                raise NotImplementedError("Sundial forward method prediction not yet implemented")
        
        # 计算均值
        mean_forecast = forecast_tensor.mean(dim=1)  # [batch_size, forecast_horizon]

        # 标准化输出形状
        # forecast_tensor = forecast_tensor.reshape(B, C, forecast_horizon, num_samples) # [batch_size, n_variates, forecast_horizon, num_samples]
        forecast_tensor = forecast_tensor.permute(0,2,1).reshape(B, C, forecast_horizon, num_samples) # [batch_size, n_variates, forecast_horizon, num_samples]
        mean_forecast = mean_forecast.reshape(B, C, forecast_horizon) # [batch_size, n_variates, forecast_horizon]
        
        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': None,
            'metadata': {
                'model': f'sundial-{self.model_type}',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon,
                'revin': revin
            }
        }
