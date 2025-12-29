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
        """提交请求到优先级队列
        
        优先级规则：数字越小优先级越高（负数 > 0 > 正数）
        例如：priority=-5 > priority=0 > priority=3
        """
        async with self._queue_lock:
            priority = request.priority
            
            if len(self.queue_list) == 0:
                # 队列为空，直接插入
                self.queue_list.append(request)
            else:
                # 计算可以超越的请求数量（优先级比当前请求低的，即数字更大的）
                can_overtake_count = sum(
                    1 for req in self.queue_list if req.priority > priority
                )
                
                if can_overtake_count > 0:
                    # 计算优先级优势比例
                    # 获取当前队列中的优先级范围
                    all_priorities = list(self.priority_distribution_cache.keys())
                    if all_priorities:
                        min_priority = min(all_priorities)  # 最高优先级（最小数字）
                        max_priority = max(all_priorities)  # 最低优先级（最大数字）
                        priority_range = max_priority - min_priority
                        
                        if priority_range > 0:
                            # 计算当前请求的优先级在范围内的相对位置
                            # 0 = 最高优先级，1 = 最低优先级
                            priority_rank_ratio = (priority - min_priority) / priority_range
                        else:
                            priority_rank_ratio = 0.5  # 所有请求优先级相同
                    else:
                        priority_rank_ratio = 0.5
                    
                    # 优先级优势 = 1 - 排名比例（越高优先级，优势越大）
                    priority_advantage = 1 - priority_rank_ratio
                    base_forward_positions = int(can_overtake_count * priority_advantage)
                    
                    max_forward_positions = min(
                        base_forward_positions * self.insert_multiplier,
                        self.max_positions,
                        can_overtake_count,
                        len(self.queue_list)
                    )
                else:
                    max_forward_positions = 0
                
                # 计算插入位置：从末尾往前数 max_forward_positions 个位置
                insert_pos = max(0, len(self.queue_list) - max_forward_positions)
                self.queue_list.insert(insert_pos, request)
                
                if max_forward_positions > 0:
                    self.logger.info(f"Priority: Request {request.request_id} (priority={priority}) "
                                   f"jumped {max_forward_positions} positions, advantage={priority_advantage:.2f}")
            
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
