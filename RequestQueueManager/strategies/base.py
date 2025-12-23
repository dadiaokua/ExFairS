"""
调度策略基类

定义所有调度策略需要实现的接口
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
import logging


@dataclass
class QueueState:
    """队列状态，传递给策略用于决策"""
    client_stats: Dict[str, Dict] = field(default_factory=dict)
    client_token_stats: Dict[str, Dict] = field(default_factory=dict)
    logger: logging.Logger = None
    
    # 通用队列
    request_queue: asyncio.Queue = None
    
    # 策略特定数据（可选）
    strategy_data: Dict[str, Any] = field(default_factory=dict)


class SchedulingStrategy(ABC):
    """
    调度策略基类
    
    所有调度策略需要实现 submit 和 get_next 方法
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """
        提交请求到队列
        
        Args:
            request: QueuedRequest 对象
            queue_state: 当前队列状态
            
        Returns:
            request_id: 请求ID
        """
        pass
    
    @abstractmethod
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """
        获取下一个要处理的请求
        
        Args:
            queue_state: 当前队列状态
            
        Returns:
            QueuedRequest 或 None（如果队列为空）
        """
        pass
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return 0
    
    async def on_task_start(self, request_id: str):
        """任务开始执行时的回调（可选实现）"""
        pass
    
    async def on_task_finish(self, request_id: str):
        """任务完成时的回调（可选实现）"""
        pass
