"""
Justitia 调度策略

基于虚拟时间的公平调度，使用最小堆维护虚拟完成时间
短任务会被优先调度，同时避免长任务饥饿
"""
import asyncio
import heapq
from typing import Optional, Any, Set, List

from .base import SchedulingStrategy, QueueState
from ..constants import DEFAULT_TOTAL_MEMORY


class JustitiaStrategy(SchedulingStrategy):
    """
    Justitia 调度策略
    
    核心思想:
    1. 虚拟时间 V(t) = M / N_t (M=总显存, N_t=当前运行任务数)
    2. 虚拟完成时间 f_j = V(a_j) + C_j (a_j=到达时间, C_j=资源需求)
    3. 调度规则: 始终选择 f_j 最小的任务
    """
    
    def __init__(self, total_memory: int = DEFAULT_TOTAL_MEMORY, **kwargs):
        super().__init__(**kwargs)
        self.total_memory = total_memory
        
        # 最小堆: (virtual_finish_time, request)
        self.heap: List[tuple] = []
        self._heap_lock = asyncio.Lock()
        
        # 正在执行的任务 ID 集合（用于准确计算虚拟时间）
        self.running_tasks: Set[str] = set()
        
        # 当前虚拟时间
        self.virtual_time = 0.0
    
    def _estimate_cost(self, request: Any) -> float:
        """估算任务资源需求 C_j"""
        request_content = request.request_content
        
        if isinstance(request_content, dict):
            input_tokens = request_content.get('input_tokens', 0)
            output_tokens = request_content.get('output_tokens', 0)
            # 使用 input * output + output²/2 作为近似
            return input_tokens * output_tokens + output_tokens * output_tokens / 2
        else:
            return 0
    
    def _update_virtual_time(self):
        """根据当前运行任务数更新虚拟时间"""
        running_count = len(self.running_tasks)
        if running_count > 0:
            self.virtual_time = self.total_memory / running_count
        else:
            self.virtual_time = self.total_memory
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到 Justitia 队列"""
        async with self._heap_lock:
            # 基于正在执行的任务数计算虚拟时间
            self._update_virtual_time()
            
            # 估算任务资源需求
            estimated_cost = self._estimate_cost(request)
            
            # 计算虚拟完成时间 f_j = V(a_j) + C_j
            virtual_finish_time = self.virtual_time + estimated_cost
            
            # 插入最小堆
            heapq.heappush(self.heap, (virtual_finish_time, request))
            
            self.logger.info(f"[Justitia] Request {request.request_id}: "
                           f"running_tasks={len(self.running_tasks)}, V(t)={self.virtual_time:.2f}, "
                           f"C_j={estimated_cost}, f_j={virtual_finish_time:.2f}, heap_size={len(self.heap)}")
        
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """获取虚拟完成时间最小的请求"""
        async with self._heap_lock:
            if not self.heap:
                return None
            
            virtual_finish_time, request = heapq.heappop(self.heap)
            
            self.logger.info(f"[Justitia] Selected request {request.request_id}, "
                           f"f_j={virtual_finish_time:.2f}, remaining={len(self.heap)}")
            
            return request
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self.heap)
    
    async def on_task_start(self, request_id: str):
        """任务开始执行时加入 running_tasks"""
        async with self._heap_lock:
            self.running_tasks.add(request_id)
            self._update_virtual_time()
            self.logger.debug(f"[Justitia] Task {request_id} started, "
                            f"running_tasks={len(self.running_tasks)}, V(t)={self.virtual_time:.2f}")
    
    async def on_task_finish(self, request_id: str):
        """任务完成时移除 running_tasks 并更新虚拟时间"""
        async with self._heap_lock:
            self.running_tasks.discard(request_id)
            self._update_virtual_time()
            self.logger.debug(f"[Justitia] Task {request_id} finished, "
                            f"running_tasks={len(self.running_tasks)}, V(t)={self.virtual_time:.2f}")
