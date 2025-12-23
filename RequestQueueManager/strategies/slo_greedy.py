"""
SLO-Greedy 调度策略

优先调度 SLO 违约率高的客户端，帮助"受苦"用户优先获得服务
包含冷启动处理，避免新客户端永远排在最后
"""
import asyncio
import heapq
import time
from typing import Optional, Any, List

from .base import SchedulingStrategy, QueueState
from ..constants import COLD_START_THRESHOLD, COLD_START_WEIGHT


class SLOGreedyStrategy(SchedulingStrategy):
    """
    SLO-Greedy 调度策略
    
    核心思想:
    1. 计算每个客户端的 SLO 违约率
    2. 优先调度违约率最高的客户端
    3. 新客户端给予初始权重避免冷启动问题
    """
    
    def __init__(self, cold_start_threshold: int = COLD_START_THRESHOLD,
                 cold_start_weight: float = COLD_START_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.cold_start_threshold = cold_start_threshold
        self.cold_start_weight = cold_start_weight
        
        # 最小堆: (-violation_rate, submit_time, request)
        # 使用负数是因为 heapq 是最小堆，我们要 violation_rate 高的优先
        self.heap: List[tuple] = []
        self._heap_lock = asyncio.Lock()
    
    def _get_violation_rate(self, client_id: str, queue_state: QueueState) -> float:
        """获取客户端的 SLO 违约率，处理冷启动"""
        client_stats = queue_state.client_stats.get(client_id, {})
        total_requests = client_stats.get('total_requests', 0)
        slo_violations = client_stats.get('failed_requests', 0)
        
        # 冷启动处理：新客户端使用初始权重
        if total_requests < self.cold_start_threshold:
            self.logger.debug(f"[SLO Greedy] Client {client_id} in cold start phase "
                            f"(requests={total_requests}), using weight={self.cold_start_weight}")
            return self.cold_start_weight
        else:
            return slo_violations / total_requests
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到 SLO-Greedy 队列"""
        async with self._heap_lock:
            violation_rate = self._get_violation_rate(request.client_id, queue_state)
            
            # 使用负的 violation_rate，因为 heapq 是最小堆
            priority_key = -violation_rate
            
            heapq.heappush(self.heap, (priority_key, time.time(), request))
            
            total_requests = queue_state.client_stats.get(request.client_id, {}).get('total_requests', 0)
            self.logger.info(f"[SLO Greedy] Request {request.request_id} from {request.client_id}: "
                           f"violation_rate={violation_rate:.3f}, total_requests={total_requests}, "
                           f"heap_size={len(self.heap)}")
        
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """获取 SLO 违约率最高的客户端的请求"""
        async with self._heap_lock:
            if not self.heap:
                return None
            
            neg_violation_rate, submit_time, request = heapq.heappop(self.heap)
            violation_rate = -neg_violation_rate
            
            self.logger.info(f"[SLO Greedy] Selected request {request.request_id} from {request.client_id}, "
                           f"violation_rate={violation_rate:.3f}, remaining={len(self.heap)}")
            
            return request
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self.heap)
