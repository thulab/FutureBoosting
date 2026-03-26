"""
时序模型基类接口
"""
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import torch
import numpy as np


class BaseTimeSeriesModel(ABC):
    """
    时序模型基类，定义统一的接口规范
    
    所有时序模型适配器都应该继承此类并实现相应的方法
    """
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化模型
        
        Args:
            model_name: 模型名称（如 'chronos2', 'moirai2' 等）
            model_path: 模型路径（Huggingface 模型ID或本地路径）
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self):
        """
        加载模型
        
        子类需要实现此方法来加载具体的模型
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        零样本推理接口
        
        Args:
            context: 输入序列，shape 为 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度（未来时间步数）
            num_samples: 采样次数（用于生成多个预测样本，用于不确定性估计）
            **kwargs: 其他模型特定的参数
        
        Returns:
            dict: 包含以下键的字典
                - 'forecast': torch.Tensor, shape [batch_size, num_samples, forecast_horizon] - 预测值
                - 'mean': torch.Tensor, shape [batch_size, forecast_horizon] - 预测均值
                - 'quantiles': Optional[torch.Tensor], shape [batch_size, num_quantiles, forecast_horizon] - 分位数预测（如果有）
                - 'metadata': dict - 其他元数据信息
        """
        pass
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True
    
    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将输入转换为torch.Tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype=dtype)
        elif isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
    
    def _normalize_input(self, context: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        标准化输入格式
        
        确保输入是 [batch_size, seq_len] 格式的torch.Tensor
        chronos2 的推理接口要求三维输入，即 [batch_size, n_variates, seq_len]，会特殊一些，不能做这个处理
        """
        context = self._to_tensor(context)
        
        # 如果是1D，添加batch维度
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        # 确保是2D [batch_size, seq_len]
        if context.dim() != 2:
            raise ValueError(f"Expected 1D or 2D input, got {context.dim()}D")
        
        return context.to(self.device)

    def _normalize_input_dim_2D(self, context: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        标准化输入格式
        
        确保输入是 [batch_size, seq_len] 格式的torch.Tensor
        """
        context = self._to_tensor(context) # 
        
        # 如果是1D，添加batch维度 [seq_len] -> [batch_size, seq_len]
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        if context.dim() == 3:
            context = context.reshape(-1, context.shape[-1]) # [batch_size, n_variates, seq_len] -> [batch_size * n_variates, seq_len]
        
        # 确保是2D [batch_size, seq_len]
        if context.dim() != 2:
            raise ValueError(f"Expected 2D input [batch_size, seq_len], got {context.dim()}D: {context.shape}")
        
        return context.to(self.device)

    def _normalize_input_dim_3D(self, context: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        标准化输入格式
        
        确保输入是 [batch_size, n_variates, seq_len] 格式的torch.Tensor
        """
        context = self._to_tensor(context)
        
        # 如果是1D，添加batch维度 [seq_len] -> [batch_size(1), n_variates(1),seq_len]
        if context.dim() == 1:
            context = context.reshape(1, 1, context.shape[-1])
        
        if context.dim() == 2: # [batch_size, seq_len] -> [batch_size, n_variates(1), seq_len]
            context = context.reshape(context.shape[0], 1, context.shape[-1])
        
        # 确保是2D [batch_size, seq_len]
        if context.dim() != 3:
            raise ValueError(f"Expected 3D input [batch_size, n_variates, seq_len], got {context.dim()}D: {context.shape}")
        
        return context.to(self.device)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name}, model_path={self.model_path}, device={self.device})"
