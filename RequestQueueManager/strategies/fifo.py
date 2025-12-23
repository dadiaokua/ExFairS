"""
FIFO 调度策略

先来先服务，最简单的调度策略
"""
import asyncio
from typing import Optional, Any

from .base import SchedulingStrategy, QueueState


class FIFOStrategy(SchedulingStrategy):
    """
    FIFO (First In First Out) 调度策略
    
    先提交的请求先处理，最简单的公平调度
    """
    
    def __init__(self, max_queue_size: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.queue = asyncio.Queue(maxsize=max_queue_size)
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到 FIFO 队列"""
        await self.queue.put(request)
        self.logger.debug(f"FIFO: Submitted request {request.request_id}, queue_size={self.queue.qsize()}")
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """从队列头部获取请求"""
        try:
            request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            self.logger.debug(f"FIFO: Retrieved request {request.request_id}")
            return request
        except asyncio.TimeoutError:
            return None
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return self.queue.qsize()
