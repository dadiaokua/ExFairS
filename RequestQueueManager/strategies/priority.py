"""
优先级调度策略

支持按轮次限制的优先级插队：
- 高优先级请求只能插到【本轮】其他请求前面
- 不能插到【上一轮】的请求前面
- 保证基本的 FIFO 公平性，避免队列拥堵
"""
import asyncio
import time
from typing import Optional, Any, List, Dict

from .base import SchedulingStrategy, QueueState
from ..constants import (
    DEFAULT_PRIORITY_INSERT_MULTIPLIER,
    DEFAULT_MAX_PRIORITY_POSITIONS,
    DEFAULT_MAX_PRIORITY_DELAY,
)


class PriorityStrategy(SchedulingStrategy):
    """
    优先级调度策略（按轮次限制插队）
    
    改进的插队规则：
    - 高优先级请求只能在【当前轮次】的请求中插队
    - 不能插到【之前轮次】的请求前面
    - 这样既保证高优先级优先，又避免无限积压低优先级请求
    """
    
    def __init__(self,
                 insert_multiplier: int = DEFAULT_PRIORITY_INSERT_MULTIPLIER,
                 max_positions: int = DEFAULT_MAX_PRIORITY_POSITIONS,
                 delay_enabled: bool = True,
                 max_delay: int = DEFAULT_MAX_PRIORITY_DELAY,
                 round_duration: float = 30.0,  # 轮次时长（秒），与监控间隔对齐
                 **kwargs):
        super().__init__(**kwargs)
        self.insert_multiplier = insert_multiplier
        self.max_positions = max_positions
        self.delay_enabled = delay_enabled
        self.max_delay = max_delay
        self.round_duration = round_duration
        
        # 使用列表模拟优先级队列
        self.queue_list: List[Any] = []
        self._queue_lock = asyncio.Lock()
        
        # 轮次管理
        self._start_time = time.time()
        self._current_round_id = 0
        
        # 优先级分布缓存
        self.priority_distribution_cache: Dict[int, int] = {}
    
    def _get_current_round_id(self) -> int:
        """计算当前轮次 ID"""
        elapsed = time.time() - self._start_time
        return int(elapsed / self.round_duration)
    
    def _find_round_boundary(self, round_id: int) -> int:
        """
        找到指定轮次的起始位置（第一个属于该轮次的请求索引）
        
        返回值：
        - 如果队列中有该轮次的请求，返回第一个的索引
        - 如果没有，返回队列长度（即末尾位置）
        """
        for i, req in enumerate(self.queue_list):
            if hasattr(req, 'round_id') and req.round_id >= round_id:
                return i
        return len(self.queue_list)
    
    async def submit(self, request: Any, queue_state: QueueState) -> str:
        """提交请求到优先级队列（按轮次限制插队）
        
        优先级规则：数字越小优先级越高（负数 > 0 > 正数）
        
        插队规则：
        - 只能在【当前轮次】的请求中根据优先级插队
        - 不能插到【之前轮次】的请求前面
        """
        async with self._queue_lock:
            priority = request.priority
            current_round = self._get_current_round_id()
            
            # 给请求打上轮次标记
            request.round_id = current_round
            
            if len(self.queue_list) == 0:
                # 队列为空，直接插入
                self.queue_list.append(request)
            else:
                # 找到当前轮次的起始位置（插队边界）
                round_boundary = self._find_round_boundary(current_round)
                
                # 只在 [round_boundary, queue_end] 范围内计算可超越的请求
                same_round_requests = self.queue_list[round_boundary:]
                can_overtake_count = sum(
                    1 for req in same_round_requests if req.priority > priority
                )
                
                if can_overtake_count > 0:
                    # 计算基于优先级的插队位置
                    if priority < 0:
                        # 高优先级（负数）：根据优先级绝对值计算优势
                        priority_advantage = min(1.0, abs(priority) / 50.0)
                    else:
                        # 普通优先级：较少的插队优势
                        priority_advantage = 0.3
                    
                    # 计算前进位置（在本轮范围内）
                    forward_positions = min(
                        int(can_overtake_count * priority_advantage * self.insert_multiplier),
                        self.max_positions,
                        can_overtake_count
                    )
                else:
                    forward_positions = 0
                
                # 计算最终插入位置：
                # 不能早于 round_boundary（保护上一轮的请求）
                # 在本轮范围内根据优先级前进
                insert_pos = max(
                    round_boundary,  # 不能插到上一轮前面
                    len(self.queue_list) - forward_positions  # 在本轮内插队
                )
                
                self.queue_list.insert(insert_pos, request)
                
                if forward_positions > 0:
                    self.logger.info(
                        f"Priority: Request {request.request_id} (priority={priority}, round={current_round}) "
                        f"jumped {forward_positions} positions within round (boundary={round_boundary})"
                    )
            
            # 更新优先级分布缓存
            self.priority_distribution_cache[priority] = \
                self.priority_distribution_cache.get(priority, 0) + 1
            
            self.logger.debug(f"Priority: Submitted request {request.request_id}, "
                            f"priority={priority}, round={current_round}, queue_size={len(self.queue_list)}")
        
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
    
    def set_round_duration(self, duration: float):
        """设置轮次时长（动态调整）"""
        self.round_duration = duration
        self.logger.info(f"Priority: Round duration set to {duration}s")

