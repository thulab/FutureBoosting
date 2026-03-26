"""
统一时序大模型零样本推理接口

支持以下模型：
- Sundial
- TimerXL
- Chronos2
- Moirai2
- TabPFN
- TiRex
"""

from .base import BaseTimeSeriesModel
from .factory import TimeSeriesModelFactory, list_available_models

__all__ = [
    'BaseTimeSeriesModel',
    'TimeSeriesModelFactory',
    'list_available_models',
]
