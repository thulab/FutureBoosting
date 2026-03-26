"""
TimesFM 模型适配器
"""
import warnings
import torch
import numpy as np
from typing import Union, Optional, Dict, Any, List
from ..base import BaseTimeSeriesModel


class TimesFMAdapter(BaseTimeSeriesModel):
    """
    TimesFM 模型适配器
    
    TimesFM (Time Series Foundation Model) 是 Google Research 开发的预训练模型
    支持点预测和概率预测
    
    支持的模型路径示例：
    - 默认使用预训练模型（无需指定路径）
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 TimesFM 适配器
        
        Args:
            model_path: 模型路径（TimesFM 使用默认预训练模型，通常不需要指定路径）
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        """
        super().__init__('timesfm', model_path, device)
    
    def load_model(self):
        """加载 TimesFM 模型"""
        try:
            import timesfm
            
            # TimesFM 使用默认预训练模型，无需指定路径
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_path, torch_compile=True)
            # self.model = timesfm.TimesFM.from_pretrained(self.model_path)
            

            self._is_loaded = True
            print("Loaded TimesFM model (default pretrained model)")
        except ImportError:
            raise ImportError(
                "TimesFM requires the 'timesfm' package. "
                "Install it with: pip install timesfm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM model: {e}")
    
    # def _infer_frequency(
    #     self,
    #     seq_len: int,
    #     freq: Optional[str] = None
    # ) -> int:
    #     """
    #     推断频率类别
        
    #     TimesFM 使用频率类别：
    #     - 0: 高频（小时级别或更短）
    #     - 1: 日频
    #     - 2: 周频或更长
        
    #     Args:
    #         seq_len: 序列长度
    #         freq: 频率字符串（'H'=小时, 'D'=天, 'W'=周等）
        
    #     Returns:
    #         int: 频率类别 (0, 1, 或 2)
    #     """
    #     if freq is None:
    #         # 根据序列长度推断
    #         if seq_len >= 1000:
    #             return 0  # 高频
    #         elif seq_len >= 100:
    #             return 1  # 日频
    #         else:
    #             return 2  # 周频或更长
        
    #     freq_upper = freq.upper()
    #     if freq_upper in ['H', 'T', 'S', 'MIN', 'SEC']:
    #         return 0  # 高频
    #     elif freq_upper in ['D', 'DAY', 'DAILY']:
    #         return 1  # 日频
    #     else:
    #         return 2  # 周频或更长
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        freq: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 TimesFM 进行零样本时序预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度（未来时间步数）
            num_samples: 采样次数（TimesFM 可能支持多样本预测）
            freq: 频率字符串（'H'=小时, 'D'=天, 'W'=周等），用于推断频率类别
            **kwargs: 其他参数，传递给 TimesFM.forecast
        
        Returns:
            dict: 预测结果，包含：
                - 'forecast': torch.Tensor, shape [batch_size, num_samples, forecast_horizon]
                - 'mean': torch.Tensor, shape [batch_size, forecast_horizon]
                - 'quantiles': Optional[torch.Tensor] - 分位数预测（如果有）
                - 'metadata': dict - 元数据信息
        """
        if num_samples != 10:
            warnings.warn(f"TimesFM is restricted to sample 10 times as its default setting (relevant to module output dimensions), setting num_samples to 10")
            num_samples = 10
        self._ensure_model_loaded()
        
        # 标准化输入格式
        B, C, L = context.shape
        context = self._normalize_input_dim_2D(context)
        batch_size = context.shape[0]
        
        # 转换为 numpy 数组
        context_np = context.cpu().numpy()

        import timesfm
        self.model.compile(
                timesfm.ForecastConfig(
                    max_context=L,
                    max_horizon=forecast_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )

        point_forecast, quantile_forecast = self.model.forecast(
            horizon=forecast_horizon,
            inputs = [context_np[i] for i in range(batch_size)]
        )
        forecast_tensor = torch.from_numpy(quantile_forecast).float().to(self.device).reshape(B, C, forecast_horizon, num_samples)
        mean_forecast = torch.from_numpy(point_forecast).float().to(self.device).reshape(B, C, forecast_horizon)
        quantile_forecast_tensor = torch.from_numpy(quantile_forecast).float().to(self.device)

        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': quantile_forecast_tensor,
            'metadata': {
                'model': 'timesfm',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon,
            }
        }
