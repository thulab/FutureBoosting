"""
时序模型工厂类，用于创建和管理不同的时序模型实例
"""
from typing import Optional, Dict, List
from .base import BaseTimeSeriesModel
from .adapters.chronos2_adapter import Chronos2Adapter
from .adapters.moirai2_adapter import Moirai2Adapter
from .adapters.timerxl_adapter import TimerXLAdapter
from .adapters.tabpfn_adapter import TabPFNAdapter
from .adapters.tirex_adapter import TiRexAdapter
from .adapters.sundial_adapter import SundialAdapter
from .adapters.timesfm_adapter import TimesFMAdapter


# 模型名称到适配器类的映射
MODEL_ADAPTERS: Dict[str, type] = {
    'chronos2': Chronos2Adapter,
    'moirai2': Moirai2Adapter,
    'timerxl': TimerXLAdapter,
    'tabpfn': TabPFNAdapter,
    'tirex': TiRexAdapter,
    'sundial': SundialAdapter,
    'timesfm': TimesFMAdapter,
}


class TimeSeriesModelFactory:
    """
    时序模型工厂类
    
    用于创建和管理不同的时序模型实例
    """
    
    @staticmethod
    def create_model(
        model_name: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseTimeSeriesModel:
        """
        创建时序模型实例
        
        Args:
            model_name: 模型名称，支持的值：
                - 'chronos2': Chronos2 模型
                - 'moirai2': Moirai2 模型
                - 'timerxl': TimerXL 模型
                - 'tabpfn': TabPFN 模型
                - 'tirex': TiRex 模型
                - 'sundial': Sundial 模型
                - 'timesfm': TimesFM 模型
            model_path: 模型路径（Huggingface 模型ID或本地路径），如果为None则使用默认路径
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
            **kwargs: 其他模型特定的参数
                - 对于 Sundial: model_type (str) - 模型类型
        
        Returns:
            BaseTimeSeriesModel: 模型实例
        
        Examples:
            >>> # 创建 Chronos2 模型
            >>> model = TimeSeriesModelFactory.create_model('chronos2')
            >>> 
            >>> # 创建 Moirai2 模型并指定路径
            >>> model = TimeSeriesModelFactory.create_model(
            ...     'moirai2',
            ...     model_path='salesforce/moirai-2.0-large'
            ... )
            >>> 
            >>> # 创建 Sundial 模型
            >>> model = TimeSeriesModelFactory.create_model(
            ...     'sundial',
            ...     model_path='/path/to/sundial',
            ...     model_type='sundial_cora'
            ... )
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in MODEL_ADAPTERS:
            available_models = ', '.join(MODEL_ADAPTERS.keys())
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {available_models}"
            )
        
        adapter_class = MODEL_ADAPTERS[model_name_lower]
        
        # 对于 Sundial，需要传递 model_type 参数
        if model_name_lower == 'sundial':
            return adapter_class(model_path=model_path, device=device, **kwargs)
        else:
            return adapter_class(model_path=model_path, device=device)
    
    @staticmethod
    def list_available_models() -> List[str]:
        """
        列出所有可用的模型名称
        
        Returns:
            List[str]: 可用的模型名称列表
        """
        return list(MODEL_ADAPTERS.keys())
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            dict: 模型信息
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in MODEL_ADAPTERS:
            raise ValueError(f"Unknown model name: {model_name}")
        
        adapter_class = MODEL_ADAPTERS[model_name_lower]
        
        info = {
            'name': model_name_lower,
            'adapter_class': adapter_class.__name__,
            'module': adapter_class.__module__,
        }
        
        # 添加默认路径信息（如果适配器有）
        try:
            instance = adapter_class()
            if hasattr(instance, 'model_path') and instance.model_path:
                info['default_path'] = instance.model_path
        except:
            pass
        
        return info


def list_available_models() -> List[str]:
    """
    便捷函数：列出所有可用的模型名称
    
    Returns:
        List[str]: 可用的模型名称列表
    """
    return TimeSeriesModelFactory.list_available_models()
