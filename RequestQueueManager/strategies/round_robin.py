"""
Round-Robin 调度策略

轮流处理不同客户端的请求，确保每个客户端都有被服务的机会
"""
import asyncio
from typing import Optional, Any, Dict, List

from .base import SchedulingStrategy, QueueState


class RoundRobinStrategy(SchedulingStrategy):
    """
    Round-Robin 调度策略
    
    按客户端轮流调度，保证每个客户端公平地获得服务机会
    """
    
    def __init__(self, max_queue_size_per_client: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_queue_size_per_client = max_queue_size_per_client
        
        # 每个客户端一个队列
        self.client_queues: Dict[str, asyncio.Queue] = {}
        self._queues_lock = asyncio.Lock()
        
        # 轮询索引
        self.round_robin_index = 0
        
        # 客户端列表（保持顺序）
        self._client_list: List[str] = []
    
    async def _ensure_client_queue(self, client_id: str):
        """确保客户端队列存在"""
        if client_id not in self.client_queues:
            async with self._queues_lock:
                if client_id not in self.client_queues:
                    self.client_queues[client_id] = asyncio.Queue(maxsize=self.max_queue_size_per_client)
                    self._client_list.append(client_id)
                    self.logger.debug(f"Round-Robin: Created queue for client {client_id}")
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到对应客户端的队列"""
        await self._ensure_client_queue(request.client_id)
        
        await self.client_queues[request.client_id].put(request)
        
        self.logger.debug(f"Round-Robin: Submitted request {request.request_id} to {request.client_id}, "
                        f"queue_size={self.client_queues[request.client_id].qsize()}")
        
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """轮流从各客户端队列获取请求"""
        if not self._client_list:
            return None
        
        # 尝试轮询所有客户端，最多尝试一轮
        attempts = 0
        max_attempts = len(self._client_list)
        
        while attempts < max_attempts:
            client_id = self._client_list[self.round_robin_index % len(self._client_list)]
            self.round_robin_index += 1
            attempts += 1
            
            # 检查该客户端是否有请求
            if client_id in self.client_queues and not self.client_queues[client_id].empty():
                try:
                    request = await asyncio.wait_for(
                        self.client_queues[client_id].get(),
                        timeout=0.1
                    )
                    self.logger.info(f"Round-Robin: Selected request {request.request_id} from {client_id}")
                    return request
                except asyncio.TimeoutError:
                    continue
        
        return None
    
    def get_queue_size(self) -> int:
        """获取所有客户端队列的总大小"""
        return sum(q.qsize() for q in self.client_queues.values())
