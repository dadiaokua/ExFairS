"""
VTC (Variable Token Credits) 调度策略

选择已消耗 token 最少的客户端的请求进行处理
使用优先队列实现 O(log n) 复杂度
"""
import asyncio
import heapq
from typing import Optional, Any, Dict, List

from .base import SchedulingStrategy, QueueState


class VTCStrategy(SchedulingStrategy):
    """
    VTC 调度策略
    
    优先调度已消耗 token 最少的客户端，实现 token 层面的公平性
    使用最小堆优化性能，从 O(n) 降为 O(log n)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 最小堆: (client_tokens, submit_time, request)
        self.heap: List[tuple] = []
        self._heap_lock = asyncio.Lock()
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到 VTC 队列"""
        async with self._heap_lock:
            # 获取客户端当前的 token 消耗
            client_tokens = queue_state.client_token_stats.get(
                request.client_id, {}
            ).get('actual_tokens_used', 0)
            
            # 插入最小堆，按 (token数, 提交时间) 排序
            heapq.heappush(self.heap, (client_tokens, request.submit_time, request))
            
            self.logger.debug(f"VTC: Submitted request {request.request_id} from {request.client_id}, "
                            f"client_tokens={client_tokens}, heap_size={len(self.heap)}")
        
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """获取 token 消耗最少的客户端的请求"""
        async with self._heap_lock:
            if not self.heap:
                return None
            
            # 从最小堆取出 token 最少的请求
            client_tokens, submit_time, request = heapq.heappop(self.heap)
            
            self.logger.info(f"VTC: Selected request {request.request_id} from {request.client_id}, "
                           f"client_tokens={client_tokens}, remaining={len(self.heap)}")
            
            return request
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self.heap)
    
    async def update_client_tokens(self, client_id: str, new_token_count: int, queue_state: QueueState):
        """
        更新客户端的 token 统计
        
        注意：由于堆中已有的元素无法直接更新优先级，
        新提交的请求会使用最新的 token 统计
        """
        if client_id in queue_state.client_token_stats:
            queue_state.client_token_stats[client_id]['actual_tokens_used'] = new_token_count
