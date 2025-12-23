"""
优先级调度策略

支持部分优先级插队，高优先级请求可以插到队列前面
"""
import asyncio
from typing import Optional, Any, List, Dict

from .base import SchedulingStrategy, QueueState
from ..constants import (
    DEFAULT_PRIORITY_INSERT_MULTIPLIER,
    DEFAULT_MAX_PRIORITY_POSITIONS,
    DEFAULT_MAX_PRIORITY_DELAY,
)


class PriorityStrategy(SchedulingStrategy):
    """
    优先级调度策略
    
    支持部分优先级插队：
    - 优先级 N 的请求可以往前插 N * multiplier 个位置
    - 有最大插队位置限制
    """
    
    def __init__(self,
                 insert_multiplier: int = DEFAULT_PRIORITY_INSERT_MULTIPLIER,
                 max_positions: int = DEFAULT_MAX_PRIORITY_POSITIONS,
                 delay_enabled: bool = True,
                 max_delay: int = DEFAULT_MAX_PRIORITY_DELAY,
                 **kwargs):
        super().__init__(**kwargs)
        self.insert_multiplier = insert_multiplier
        self.max_positions = max_positions
        self.delay_enabled = delay_enabled
        self.max_delay = max_delay
        
        # 使用列表模拟优先级队列
        self.queue_list: List[Any] = []
        self._queue_lock = asyncio.Lock()
        
        # 优先级分布缓存
        self.priority_distribution_cache: Dict[int, int] = {}
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到优先级队列"""
        async with self._queue_lock:
            priority = request.priority
            
            if priority <= 0 or len(self.queue_list) == 0:
                # 优先级0或队列为空，直接插入末尾
                self.queue_list.append(request)
            else:
                # 计算可以插队的位置
                can_overtake_count = sum(
                    1 for req in self.queue_list if req.priority < priority
                )
                
                if can_overtake_count > 0:
                    # 计算优先级优势比例
                    priority_rank_ratio = priority / max(
                        max(self.priority_distribution_cache.keys(), default=1), 1
                    )
                    base_forward_positions = int(can_overtake_count * priority_rank_ratio)
                    
                    max_forward_positions = min(
                        base_forward_positions * self.insert_multiplier,
                        self.max_positions,
                        can_overtake_count,
                        len(self.queue_list)
                    )
                else:
                    max_forward_positions = 0
                
                # 计算插入位置
                insert_pos = max(0, len(self.queue_list) - max_forward_positions)
                self.queue_list.insert(insert_pos, request)
                
                if max_forward_positions > 0:
                    self.logger.info(f"Priority: Request {request.request_id} (priority={priority}) "
                                   f"jumped {max_forward_positions} positions")
            
            # 更新优先级分布缓存
            self.priority_distribution_cache[priority] = \
                self.priority_distribution_cache.get(priority, 0) + 1
            
            self.logger.debug(f"Priority: Submitted request {request.request_id}, "
                            f"priority={priority}, queue_size={len(self.queue_list)}")
        
        return request.request_id
    
    async def get_next(self, queue_state: QueueState) -> Optional[Any]:
        """从队列头部获取请求"""
        async with self._queue_lock:
            if not self.queue_list:
                return None
            
            request = self.queue_list.pop(0)
            
            # 更新优先级分布缓存
            priority = request.priority
            if priority in self.priority_distribution_cache:
                self.priority_distribution_cache[priority] -= 1
                if self.priority_distribution_cache[priority] <= 0:
                    del self.priority_distribution_cache[priority]
            
            self.logger.debug(f"Priority: Retrieved request {request.request_id}, "
                            f"priority={priority}, remaining={len(self.queue_list)}")
            
            return request
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self.queue_list)
