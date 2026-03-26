"""
TimerXL 模型适配器
"""
import warnings
import torch
import numpy as np
from typing import Union, Optional, Dict, Any
from ..base import BaseTimeSeriesModel


class TimerXLAdapter(BaseTimeSeriesModel):
    """
    TimerXL 模型适配器
    
    支持的模型路径示例：
    - timer-base-84m
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, model_type: str = 'timerxl'):
        """
        初始化 TimerXL 适配器
        
        Args:
            model_path: Huggingface 模型ID或本地路径，默认为 'autotimers/TimerXL-base'
            device: 设备
            model_type: 模型类型（'timerxl', 'timerxl-cora', 'timerxl-dualweaver'等）
        """
        self.model_type = model_type
        if model_path is None:
            model_path = 'thuml/timer-base-84m' # [qiuyz] NOTE: this is the default model path, should be changed to the actual model path
        super().__init__('timerxl', model_path, device)
    
    def load_model(self):
        """加载 TimerXL 模型"""
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            
            if self.model_path is None:
                raise ValueError("TimerXL model requires model_path to be specified")
            
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
            print(f"Loaded TimerXL ({self.model_type}) model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TimerXL model: {e}")
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        forecast_horizon: int,
        num_samples: Optional[int] = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 TimerXL 进行预测
        
        Args:
            context: 输入序列 [batch_size, seq_len] 或 [seq_len]
            forecast_horizon: 预测长度
            num_samples: 采样次数
            **kwargs: 其他参数
        
        Returns:
            dict: 预测结果
        """
        if num_samples != 1:
            warnings.warn("TimerXL is not a probablistic model, only supports num_samples=1, setting num_samples to 1")
            num_samples = 1
        self._ensure_model_loaded()
        
        B, C, L = context.shape
        context = self._normalize_input_dim_2D(context)
        zero_var_mask = self._get_zero_variance_mask(context)
        forecast_tensor = torch.empty(
            (context.shape[0], forecast_horizon),
            dtype=torch.float32,
            device=context.device,
        )
        
        with torch.no_grad():
            # TimerXL 的预测接口
            if hasattr(self.model, 'generate'):
                normal_mask = ~zero_var_mask
                if normal_mask.any():
                    forecast_tensor[normal_mask] = self._run_generate(
                        context=context[normal_mask],
                        forecast_horizon=forecast_horizon,
                        revin=True,
                    )
                if zero_var_mask.any():
                    n_zero = int(zero_var_mask.sum().item())
                    warnings.warn(
                        f"TimerXL detected {n_zero} zero-variance input series; "
                        "running them with revin=False."
                    )
                    forecast_tensor[zero_var_mask] = self._run_generate(
                        context=context[zero_var_mask],
                        forecast_horizon=forecast_horizon,
                        revin=False,
                    )
            else:
                # 使用 forward 方法（需要提供完整的输入）
                # 注意：这可能需要调整
                raise NotImplementedError("TimerXL forward method prediction not yet implemented")
        
        # 确保正确的shape
        forecast_tensor = forecast_tensor.reshape(B, C, forecast_horizon, num_samples) # [batch_size, n_variates, forecast_horizon, num_samples]
        mean_forecast = forecast_tensor.mean(dim=-1) # [batch_size, n_variates, forecast_horizon]
        
        return {
            'forecast': forecast_tensor,
            'mean': mean_forecast,
            'quantiles': None,
            'metadata': {
                'model': f'timerxl-{self.model_type}',
                'num_samples': num_samples,
                'forecast_horizon': forecast_horizon
            }
        }

    def _run_generate(
        self,
        *,
        context: torch.Tensor,
        forecast_horizon: int,
        revin: bool,
    ) -> torch.Tensor:
        outputs = self.model.generate(
            context,
            max_new_tokens=forecast_horizon,
            revin=revin,
        )
        return self._normalize_generate_output(
            outputs=outputs,
            context=context,
            forecast_horizon=forecast_horizon,
        )

    def _get_zero_variance_mask(self, context: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        std = context.std(dim=-1, keepdim=False, unbiased=False)
        return torch.isfinite(std) & (std <= float(eps))

    def _normalize_generate_output(
        self,
        *,
        outputs,
        context: torch.Tensor,
        forecast_horizon: int,
    ) -> torch.Tensor:
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(outputs, dtype=torch.float32, device=context.device)
        else:
            outputs = outputs.to(device=context.device, dtype=torch.float32)

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if outputs.dim() != 2:
            raise ValueError(f"TimerXL generate output must be 2D, got shape={tuple(outputs.shape)}")

        if outputs.shape[1] < forecast_horizon:
            raise ValueError(
                f"TimerXL generate output too short: shape={tuple(outputs.shape)}, "
                f"forecast_horizon={forecast_horizon}"
            )
        if outputs.shape[1] > forecast_horizon:
            outputs = outputs[:, -forecast_horizon:]

        bad = ~torch.isfinite(outputs)
        if bad.any():
            fallback = context[:, -1:].expand(-1, forecast_horizon)
            n_bad = int(bad.sum().item())
            n_series = int(bad.any(dim=1).sum().item())
            warnings.warn(
                f"TimerXL generated {n_bad} non-finite values across {n_series} series; "
                "replacing them with last-value fallback."
            )
            outputs = torch.where(bad, fallback, outputs)

        return outputs
