import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from util.RequestUtil import make_request
import os
from config.Config import GLOBAL_CONFIG
import json
import uuid  # 添加uuid导入
import threading  # 添加线程安全支持


# 全局计数器，用于生成唯一的请求序列号
_request_counter = 0
_counter_lock = threading.Lock()

def generate_unique_request_id(client_id: str, worker_id: str) -> str:
    """生成唯一的请求ID，避免重复"""
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        counter = _request_counter
    
    # 使用UUID确保全局唯一性，并添加可读的前缀
    unique_id = str(uuid.uuid4())[:8]  # 取UUID的前8位
    timestamp = int(time.time() * 1000000)  # 使用微秒时间戳
    
    return f"req_{client_id}_{worker_id}_{counter}_{timestamp}_{unique_id}"

class QueueStrategy(Enum):
    """队列调度策略"""
    FIFO = "fifo"  # 先进先出
    PRIORITY = "priority"  # 基于优先级
    ROUND_ROBIN = "round_robin"  # 轮询
    SHORTEST_JOB_FIRST = "sjf"  # 最短作业优先
    FAIR_SHARE = "fair_share"  # 公平共享
    VTC = "vtc"


@dataclass
class QueuedRequest:
    """队列中的请求对象"""
    start_time: float
    client_id: str
    worker_id: str
    request_content: str
    experiment: Any
    priority: int = 0
    submit_time: float = 0
    client_type: str = "unknown"  # short or long
    request_id: str = ""  # 添加request_id字段

    def __post_init__(self):
        if self.submit_time == 0:
            self.submit_time = time.time()
        # 如果没有request_id，生成一个唯一的
        if not self.request_id:
            self.request_id = generate_unique_request_id(self.client_id, self.worker_id)

    def __lt__(self, other):
        """用于优先队列排序"""
        return self.priority < other.priority


class RequestQueueManager:
    """请求队列管理器，负责控制所有客户端请求的顺序"""

    def __init__(self, strategy: QueueStrategy = QueueStrategy.FIFO, max_queue_size: int = 10000):
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.response_queues: Dict[str, asyncio.Queue] = {}  # 每个客户端的响应队列
        self.client_stats: Dict[str, Dict] = {}  # 客户端统计信息
        self.client_token_stats: Dict[str, Dict] = {}  # 每个客户端的token统计
        self.is_running = False
        self.workers_running = False
        self.openai_client = None
        self.logger = self._setup_logger()

        # 不同策略的特定数据结构
        self.priority_queue_list = []  # 改为列表，用于部分优先级
        self.priority_queue_lock = asyncio.Lock()  # 添加锁保护优先级队列
        self.round_robin_index = 0  # 轮询索引
        self.client_request_counts: Dict[str, int] = {}  # 每个客户端的请求计数
        
        # 为轮询策略维护每个客户端的队列
        self.client_queues: Dict[str, asyncio.Queue] = {}  # 每个客户端的请求队列

        # 部分优先级配置
        self.priority_insert_multiplier = 1  # 优先级倍数，优先级N可以往前插N*multiplier个位置
        self.max_priority_positions = 100  # 最大优先级插入位置限制
        self.priority_delay_enabled = True  # 是否启用低优先级延迟
        self.max_priority_delay = 10  # 最大延迟秒数

        # 优化：维护优先级分布的缓存，避免每次重新计算
        self.priority_distribution_cache = {}  # {priority: count}

        # 统计信息
        self.total_requests_processed = 0
        self.start_time = None

        # 队列监控配置
        self.queue_monitor_interval = GLOBAL_CONFIG.get("queue_monitor_interval", 5)  # 队列监控间隔（秒）
        self.queue_monitor_task = None  # 队列监控任务

        # 用于跟踪已提交的request_id，避免重复
        self._submitted_request_ids = set()

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("RequestQueueManager")
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            # 确保log目录存在
            os.makedirs('log', exist_ok=True)

            # 使用全局配置中的时间戳
            timestamp = GLOBAL_CONFIG.get("monitor_file_time", "default")

            # 创建文件处理器
            fh = logging.FileHandler(filename=f'log/request_queue_manager_{timestamp}.log', encoding="utf-8", mode="a")
            fh.setLevel(logging.DEBUG)

            # 创建控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 创建格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            # 添加处理器到日志记录器
            logger.addHandler(ch)
            logger.addHandler(fh)

            # 确保日志不会被父级处理器处理
            logger.propagate = False
        return logger

    def set_openai_client(self, client):
        """设置OpenAI客户端"""
        self.openai_client = client

    def configure_partial_priority(self, insert_multiplier: int = 3, max_positions: int = 20, 
                                 delay_enabled: bool = True, max_delay: int = 10):
        """配置部分优先级参数
        
        Args:
            insert_multiplier: 优先级倍数，优先级N可以往前插N*multiplier个位置
            max_positions: 最大优先级插入位置限制
            delay_enabled: 是否启用低优先级延迟
            max_delay: 最大延迟秒数
        """
        self.priority_insert_multiplier = insert_multiplier
        self.max_priority_positions = max_positions
        self.priority_delay_enabled = delay_enabled
        self.max_priority_delay = max_delay
        self.logger.info(f"Configured partial priority: multiplier={insert_multiplier}, max_positions={max_positions}, "
                        f"delay_enabled={delay_enabled}, max_delay={max_delay}")

    async def register_client(self, client_id: str, client_type: str = "unknown"):
        """注册客户端"""
        # 检查客户端是否已经存在，避免重复注册
        if client_id in self.response_queues:
            self.logger.debug(f"Client {client_id} already registered, skipping")
            return

        self.response_queues[client_id] = asyncio.Queue()
        self.client_stats[client_id] = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_wait_time': 0,
            'client_type': client_type
        }
        self.client_request_counts[client_id] = 0
        self.client_token_stats[client_id] = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'actual_tokens_used': 0
        }
        
        # 为轮询策略创建客户端队列
        if self.strategy == QueueStrategy.ROUND_ROBIN:
            self.client_queues[client_id] = asyncio.Queue(maxsize=self.max_queue_size // 10)  # 每个客户端队列较小
        
        self.logger.info(f"Registered client: {client_id} (type: {client_type})")
        
        # 如果是轮询策略，更新客户端列表
        if self.strategy == QueueStrategy.ROUND_ROBIN:
            self._round_robin_clients = sorted(self.client_stats.keys())

    async def _monitor_queue_status(self):
        """队列状态监控协程"""
        self.logger.info(f"[queue manager 队列监控] 监控已启动，间隔: {self.queue_monitor_interval}s")

        # 准备监控日志文件
        os.makedirs('tmp_result', exist_ok=True)
        timestamp = GLOBAL_CONFIG.get("monitor_file_time", datetime.now().strftime("%m_%d_%H_%M"))
        monitor_file = f'tmp_result/queue_monitor_{timestamp}.json'

        monitor_data = {
            'start_time': datetime.now().isoformat(),
            'monitor_interval': self.queue_monitor_interval,
            'strategy': self.strategy.value,
            'snapshots': []
        }

        while self.workers_running:
            try:
                # 获取队列状态
                normal_queue_size = self.request_queue.qsize()

                # 创建优先级队列的快照（只读操作，无需加锁）
                priority_queue_snapshot = self.priority_queue_list.copy()
                priority_queue_size = len(priority_queue_snapshot)

                # 统计不同优先级的请求数量
                priority_counts = {}
                for request in priority_queue_snapshot:
                    priority = request.priority
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1

                # 按客户端分组统计队列中的请求
                client_queue_stats = {}
                for request in priority_queue_snapshot:
                    client_id = request.client_id
                    if client_id not in client_queue_stats:
                        client_queue_stats[client_id] = {
                            'total_in_queue': 0,
                            'priority_distribution': {},
                            'requests': []
                        }

                    client_queue_stats[client_id]['total_in_queue'] += 1
                    priority = request.priority
                    if priority not in client_queue_stats[client_id]['priority_distribution']:
                        client_queue_stats[client_id]['priority_distribution'][priority] = 0
                    client_queue_stats[client_id]['priority_distribution'][priority] += 1

                    # 记录请求详情
                    wait_time = time.time() - request.submit_time
                    client_queue_stats[client_id]['requests'].append({
                        'request_id': request.request_id,
                        'priority': priority,
                        'wait_time': round(wait_time, 2),
                        'submit_time': request.submit_time,
                        'client_type': getattr(request, 'client_type', 'unknown')
                    })

                # 统计客户端处理情况
                client_processing_stats = {}
                for client_id, stats in self.client_stats.items():
                    pending_requests = stats['total_requests'] - stats['completed_requests'] - stats['failed_requests']
                    avg_wait_time = stats['total_wait_time'] / max(stats['completed_requests'], 1)
                    success_rate = stats['completed_requests'] / max(stats['total_requests'], 1) * 100 if stats['total_requests'] > 0 else 0

                    # 获取token统计
                    token_stats = self.client_token_stats.get(client_id, {})

                    client_processing_stats[client_id] = {
                        'client_type': stats['client_type'],
                        'total_requests': stats['total_requests'],
                        'completed_requests': stats['completed_requests'],
                        'failed_requests': stats['failed_requests'],
                        'pending_requests': pending_requests,
                        'success_rate': round(success_rate, 2),
                        'avg_wait_time': round(avg_wait_time, 3),
                        'total_input_tokens': token_stats.get('total_input_tokens', 0),
                        'total_output_tokens': token_stats.get('total_output_tokens', 0),
                        'actual_tokens_used': token_stats.get('actual_tokens_used', 0)
                    }

                # 计算总的待处理请求数
                total_pending = normal_queue_size + priority_queue_size

                # 创建监控快照
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'queue_summary': {
                        'total_pending': total_pending,
                        'normal_queue_size': normal_queue_size,
                        'priority_queue_size': priority_queue_size,
                        'priority_distribution': dict(sorted(priority_counts.items())) if priority_counts else {}
                    },
                    'client_queue_stats': client_queue_stats,
                    'client_processing_stats': client_processing_stats
                }

                monitor_data['snapshots'].append(snapshot)

                # 保存到文件
                with open(monitor_file, 'w', encoding='utf-8') as f:
                    json.dump(monitor_data, f, indent=2, ensure_ascii=False)

                # 打印简要日志到控制台
                priority_info = ""
                if priority_counts:
                    priority_info = f", 优先级分布: {dict(sorted(priority_counts.items()))}"

                if normal_queue_size > 100 or priority_queue_size > 100:
                    self.logger.info(
                        f"[queue manager 队列监控] 总待处理: {total_pending} (普通队列: {normal_queue_size}, 优先级队列: {priority_queue_size}{priority_info})")

                # 打印客户端摘要
                if client_queue_stats:
                    for client_id, queue_stats in client_queue_stats.items():
                        processing_stats = client_processing_stats.get(client_id, {})
                        self.logger.info(f"  客户端 {client_id}: 队列中 {queue_stats['total_in_queue']} 个请求, "
                                         f"已处理 {processing_stats.get('completed_requests', 0)}/{processing_stats.get('total_requests', 0)}, "
                                         f"成功率 {processing_stats.get('success_rate', 0):.1f}%")

            except Exception as e:
                self.logger.error(f"[queue manager 队列监控] 监控过程中出现错误: {e}")

            # 等待下一次监控
            await asyncio.sleep(self.queue_monitor_interval)

        # 保存最终统计
        monitor_data['end_time'] = datetime.now().isoformat()
        monitor_data['total_snapshots'] = len(monitor_data['snapshots'])

        try:
            with open(monitor_file, 'w', encoding='utf-8') as f:
                json.dump(monitor_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[queue manager 队列监控] 监控数据已保存到: {monitor_file}")
        except Exception as e:
            self.logger.error(f"[queue manager 队列监控] 保存监控数据失败: {e}")

        self.logger.info("[queue manager 队列监控] 监控已停止")

    async def submit_request(self, start_time: float, client_id: str, worker_id: str, request_content: str,
                             experiment: Any, priority: int = 0, request_id: str = None) -> str:
        """提交请求到队列"""
        if client_id not in self.response_queues:
            await self.register_client(client_id)

        # 如果没有提供request_id，生成一个唯一的
        if request_id is None:
            request_id = generate_unique_request_id(client_id, worker_id)

        # 验证request_id的唯一性（可选的调试检查）
        if hasattr(self, '_submitted_request_ids'):
            if request_id in self._submitted_request_ids:
                self.logger.error(f"Duplicate request_id detected: {request_id}")
                # 重新生成一个新的ID
                request_id = generate_unique_request_id(client_id, worker_id)
            self._submitted_request_ids.add(request_id)
        else:
            self._submitted_request_ids = {request_id}

        request = QueuedRequest(
            start_time=start_time,
            client_id=client_id,
            worker_id=worker_id,
            request_content=request_content,
            experiment=experiment,
            priority=priority,
            client_type=self.client_stats[client_id]['client_type'],
            request_id=request_id  # 传递request_id
        )

        # 更新客户端统计
        self.client_stats[client_id]['total_requests'] += 1
        self.client_request_counts[client_id] += 1

        try:
            if self.strategy == QueueStrategy.PRIORITY:
                # 部分优先级策略：根据优先级在整个缓存中的排名位置来决定插入策略
                # 数字越小的优先级越高，但不会直接插到所有低优先级请求的前面
                # 而是根据优先级差值计算允许的前进位置数
                
                # 低优先级请求延迟插入队列
                if self.priority_delay_enabled and request.priority > 0:
                    # 计算延迟时间：根据优先级在系统中的相对位置
                    # 获取当前系统中所有优先级
                    async with self.priority_queue_lock:
                        all_priorities = sorted(self.priority_distribution_cache.keys())
                        if len(all_priorities) > 0:
                            # 计算当前请求的优先级排名比例
                            higher_priority_count = len([p for p in all_priorities if p < request.priority])
                            total_priority_levels = len(all_priorities)
                            priority_rank_ratio = higher_priority_count / total_priority_levels
                            
                            # 根据排名比例计算延迟：排名越低（ratio越大），延迟越长
                            # 排名比例0-1，转换为延迟0-max_delay秒
                            delay_seconds = min(priority_rank_ratio * self.max_priority_delay, self.max_priority_delay)
                            
                            self.logger.info(f"低优先级请求 {request.request_id} (priority={request.priority}) "
                                           f"排名比例={priority_rank_ratio:.3f}, 延迟 {delay_seconds:.1f} 秒后插入队列")
                        else:
                            # 如果系统中没有其他优先级，使用默认延迟
                            delay_seconds = min(request.priority, self.max_priority_delay)
                            self.logger.info(f"低优先级请求 {request.request_id} (priority={request.priority}) "
                                           f"系统无其他优先级，延迟 {delay_seconds} 秒后插入队列")
                    
                    await asyncio.sleep(delay_seconds)
                
                async with self.priority_queue_lock:
                    if len(self.priority_distribution_cache) == 0:
                        # 缓存为空，直接插入到末尾
                        insert_pos = len(self.priority_queue_list)
                    else:
                        # 获取所有优先级并排序（数值越小优先级越高）
                        all_priorities = sorted(self.priority_distribution_cache.keys())

                        # 计算当前请求的优先级排名
                        # 找到比当前优先级更高（数值更小）的优先级数量
                        higher_priority_count = len([p for p in all_priorities if p < request.priority])
                        total_priority_levels = len(all_priorities)

                        # 计算优先级排名比例（0表示最高优先级，1表示最低优先级）
                        if request.priority in all_priorities:
                            # 如果当前优先级已存在，使用现有排名
                            priority_rank_ratio = higher_priority_count / total_priority_levels
                        else:
                            # 如果是新的优先级，计算插入后的排名
                            priority_rank_ratio = higher_priority_count / (total_priority_levels + 1)

                        # 计算可以超越的请求数量（比当前优先级低的所有请求）
                        can_overtake_count = 0
                        for existing_priority, count in self.priority_distribution_cache.items():
                            if existing_priority > request.priority:
                                can_overtake_count += count

                        # 根据优先级排名比例计算可以往前插入的位置数
                        if can_overtake_count > 0:
                            # 优先级排名越高（ratio越小），可以往前插入的比例越大
                            # 使用反比例：(1 - priority_rank_ratio) 表示优先级优势
                            priority_advantage = 1 - priority_rank_ratio

                            # 计算基础前进位置数
                            base_forward_positions = int(can_overtake_count * priority_advantage)

                            # 应用倍数和限制
                            max_forward_positions = min(
                                base_forward_positions * self.priority_insert_multiplier,
                                self.max_priority_positions,
                                can_overtake_count,
                                len(self.priority_queue_list)
                            )
                        else:
                            max_forward_positions = 0

                        # 计算实际插入位置：从末尾往前数 max_forward_positions 个位置
                        insert_pos = max(0, len(self.priority_queue_list) - max_forward_positions)

                        self.logger.info(f"优先级请求 {request.request_id} (priority={request.priority}): "
                                          f"排名比例={priority_rank_ratio:.3f}, 可超越={can_overtake_count}, "
                                          f"前进位置={max_forward_positions}, 插入位置={insert_pos}, "
                                          f"队列总长度={len(self.priority_queue_list)}")

                    # 执行插入操作
                    self.priority_queue_list.insert(insert_pos, request)

                    # 更新优先级分布缓存（增量更新）
                    self.priority_distribution_cache[request.priority] = self.priority_distribution_cache.get(
                        request.priority, 0) + 1

                    # 记录插入统计
                    if insert_pos < len(self.priority_queue_list):
                        jumped_positions = len(self.priority_queue_list) - insert_pos
                        self.logger.info(f"请求 {request.request_id} (priority={request.priority}) "
                                         f"在队列中前进了 {jumped_positions} 个位置 (队列总长度: {len(self.priority_queue_list)})")
                    else:
                        self.logger.debug(f"请求 {request.request_id} (priority={request.priority}) "
                                          f"插入到队列末尾 (队列总长度: {len(self.priority_queue_list)})")
            elif self.strategy == QueueStrategy.ROUND_ROBIN:
                # 轮询策略：将请求放入对应客户端的队列
                if client_id not in self.client_queues:
                    # 如果客户端队列不存在，创建一个
                    self.client_queues[client_id] = asyncio.Queue(maxsize=self.max_queue_size // 10)
                
                await self.client_queues[client_id].put(request)
                self.logger.debug(f"Submitted request from {client_id} to client queue (request_id: {request_id})")
            else:
                # 其他策略使用普通队列
                await self.request_queue.put(request)
                self.logger.debug(f"Submitted request from {client_id} to queue (request_id: {request_id})")

            return request_id  # 返回传入的或生成的request_id
        except asyncio.QueueFull:
            self.logger.error(f"Queue is full, rejecting request from {client_id}")
            self.client_stats[client_id]['failed_requests'] += 1
            raise Exception("Request queue is full")

    async def get_response(self, client_id: str, timeout: float = 30.0) -> Optional[Any]:
        """获取客户端的响应"""
        if client_id not in self.response_queues:
            return None

        try:
            response = await asyncio.wait_for(
                self.response_queues[client_id].get(),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"Response timeout for client {client_id}")
            return None

    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """根据策略获取下一个请求"""
        if self.strategy == QueueStrategy.FIFO:
            return await self._get_fifo_request()
        elif self.strategy == QueueStrategy.PRIORITY:
            return await self._get_priority_request()
        elif self.strategy == QueueStrategy.ROUND_ROBIN:
            return await self._get_round_robin_request()
        elif self.strategy == QueueStrategy.SHORTEST_JOB_FIRST:
            return await self._get_sjf_request()
        elif self.strategy == QueueStrategy.FAIR_SHARE:
            return await self._get_fair_share_request()
        elif self.strategy == QueueStrategy.VTC:
            return await self._get_vtc_request()
        else:
            return await self._get_fifo_request()

    async def _get_fifo_request(self) -> Optional[QueuedRequest]:
        """FIFO策略"""
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _get_priority_request(self) -> Optional[QueuedRequest]:
        """部分优先级策略：从列表头部取出请求"""
        async with self.priority_queue_lock:
            if self.priority_queue_list:
                request = self.priority_queue_list.pop(0)  # 从头部取出（FIFO基础上的部分优先级）

                # 更新优先级分布缓存（减量更新）
                if request.priority in self.priority_distribution_cache:
                    self.priority_distribution_cache[request.priority] -= 1
                    if self.priority_distribution_cache[request.priority] <= 0:
                        del self.priority_distribution_cache[request.priority]

                return request
            return None

    async def _get_round_robin_request(self) -> Optional[QueuedRequest]:
        """轮询策略：轮流处理不同客户端的请求"""
        if not self.client_queues:
            return None

        # 使用固定的客户端列表（按注册顺序排序）
        all_clients = sorted(self.client_queues.keys())
        if not all_clients:
            return None
        
        # 尝试轮询所有客户端，最多尝试一轮
        attempts = 0
        max_attempts = len(all_clients)
        
        while attempts < max_attempts:
            # 使用固定客户端列表进行轮询
            target_client_id = all_clients[self.round_robin_index % len(all_clients)]
            self.round_robin_index += 1
            attempts += 1
            
            # 检查该客户端是否有请求
            if target_client_id in self.client_queues and not self.client_queues[target_client_id].empty():
                try:
                    request = await asyncio.wait_for(
                        self.client_queues[target_client_id].get(), 
                        timeout=0.01
                    )
                    self.logger.debug(f"Round-robin selected request from client {target_client_id} "
                                     f"(client index: {(self.round_robin_index - 1) % len(all_clients)})")
                    return request
                except asyncio.TimeoutError:
                    # 队列在获取时变空，继续下一个客户端
                    continue
            # 如果该客户端没有请求，直接跳到下一个客户端（保持轮询顺序）
        
        # 所有客户端都没有请求
        return None

    async def _get_sjf_request(self) -> Optional[QueuedRequest]:
        """最短作业优先策略"""
        # 需要收集一些请求后按estimated_tokens排序
        # 这里简化为FIFO
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _get_fair_share_request(self) -> Optional[QueuedRequest]:
        """公平共享策略"""
        # 简化实现：优先处理请求数较少的客户端
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _get_vtc_request(self) -> Optional[QueuedRequest]:
        """VTC策略：选择actual_tokens_used最小的客户端的最早请求"""
        if self.request_queue.qsize() == 0:
            return None

        # 收集所有请求
        all_requests = []
        temp_queue_size = self.request_queue.qsize()

        for _ in range(temp_queue_size):
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                all_requests.append(request)
            except asyncio.TimeoutError:
                break

        if not all_requests:
            return None

        # 找到tokens最少的请求
        min_tokens_request = None
        min_tokens = float('inf')
        min_submit_time = float('inf')

        for request in all_requests:
            client_tokens = self.client_token_stats.get(request.client_id, {}).get('actual_tokens_used', 0)

            # 优先选择tokens最少的，如果tokens相同则选择提交时间最早的
            if (client_tokens < min_tokens or
                    (client_tokens == min_tokens and request.submit_time < min_submit_time)):
                min_tokens = client_tokens
                min_submit_time = request.submit_time
                min_tokens_request = request

        # 把除了选中请求外的其他请求放回队列
        for request in all_requests:
            if request != min_tokens_request:
                await self.request_queue.put(request)

        if min_tokens_request:
            self.logger.debug(f"VTC selected request from {min_tokens_request.client_id} (tokens: {min_tokens})")

        return min_tokens_request

    async def _process_request(self, request: QueuedRequest, worker_name) -> Any:
        """处理单个请求"""
        self.logger.debug(f"Worker {worker_name}: Starting to process request {request.request_id}")

        if not self.openai_client:
            self.logger.error("OpenAI client not set")
            return None
        else:
            selected_client = self.openai_client[int(worker_name.split('-')[1]) % len(self.openai_client)]
            self.logger.debug(
                f"Worker {worker_name}: Selected client {type(selected_client)} for request {request.request_id}")

        wait_time = time.time() - request.submit_time
        self.client_stats[request.client_id]['total_wait_time'] += wait_time
        self.logger.debug(f"Worker {worker_name}: Request {request.request_id} waited {wait_time:.3f} seconds")

        try:
            self.logger.debug(f"Worker {worker_name}: Calling make_request for {request.request_id}")
            # 调用原有的make_request函数，传递request_id
            result = await make_request(
                openai=selected_client,
                experiment=request.experiment,
                request=request.request_content,
                start_time=request.start_time,
                request_id=request.request_id,  # 传递request_id
                priority=request.priority
            )

            if result is None:
                self.client_stats[request.client_id]['failed_requests'] += 1
                self.logger.warning(f"Request failed: {request.request_id}")
                return None

            self.client_stats[request.client_id]['completed_requests'] += 1
            self.total_requests_processed += 1
            self.logger.debug(f"Request completed successfully: {request.request_id}")

            try:
                # 从result中提取token信息并更新统计
                # result格式: (output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count, slo_compliance)
                if isinstance(result, (tuple, list)) and len(result) >= 6:  # 确保有6个返回值
                    output_tokens = int(result[0])
                    input_token_count = int(result[4])

                    self.client_token_stats[request.client_id]['total_output_tokens'] += output_tokens
                    self.client_token_stats[request.client_id]['total_input_tokens'] += input_token_count
                    self.client_token_stats[request.client_id]['actual_tokens_used'] += (
                                output_tokens + input_token_count)
            except (ValueError, TypeError, IndexError) as e:
                self.logger.error(f"Error processing result data for {request.request_id}: {str(e)}, result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id} from {request.client_id}: {str(e)}")
            self.client_stats[request.client_id]['failed_requests'] += 1
            return None

    async def start_processing(self, num_workers: int = 5):
        """启动请求处理"""
        if self.is_running:
            self.logger.warning("Queue manager is already running")
            return

        self.is_running = True
        self.workers_running = True
        self.start_time = time.time()

        self.logger.info(f"Starting request queue manager with {num_workers} workers")
        self.logger.info(f"Strategy: {self.strategy}, OpenAI client set: {self.openai_client is not None}")

        if self.openai_client is None:
            self.logger.error("CRITICAL: OpenAI client is not set! Workers will not be able to process requests.")
        else:
            self.logger.info(f"OpenAI client configured with {len(self.openai_client)} clients")

        # 创建工作协程
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]

        self.logger.info(f"Created {len(workers)} worker tasks")

        # 启动队列监控协程
        self.queue_monitor_task = asyncio.create_task(self._monitor_queue_status())
        self.logger.info("Started queue monitoring task")

        try:
            # 持续运行直到被明确停止
            # 使用一个无限循环来保持队列管理器运行
            while self.workers_running:
                # 检查worker状态，如果有worker意外停止，重新启动它们
                for i, worker in enumerate(workers):
                    if worker.done():
                        self.logger.warning(f"Worker worker-{i} stopped unexpectedly, restarting...")
                        workers[i] = asyncio.create_task(self._worker(f"worker-{i}"))

                # 短暂等待后再次检查
                await asyncio.sleep(1.0)

        except Exception as e:
            self.logger.error(f"Error in queue processing: {e}")
        finally:
            # 只有在明确停止时才设置为False
            if not self.workers_running:
                self.is_running = False

                # 取消所有worker
                for worker in workers:
                    if not worker.done():
                        worker.cancel()

                # 等待所有worker完成取消
                await asyncio.gather(*workers, return_exceptions=True)

                self.logger.info("Queue manager processing stopped")

    async def _worker(self, worker_name: str):
        """工作协程 - 每隔3秒取出所有请求并依次处理"""
        self.logger.info(f"Worker {worker_name} started (batch mode: 3s interval)")

        consecutive_empty_cycles = 0
        max_empty_cycles = 1200  # 120秒无请求后记录警告，但不退出

        while self.workers_running:
            try:
                # 每隔1秒处理一批请求
                await asyncio.sleep(GLOBAL_CONFIG.get('queue_worker_sleep_time', 1))
                
                # 收集所有可用的请求
                requests_to_process = []
                
                # 先计算有多少个请求
                normal_queue_size = self.request_queue.qsize()
                priority_queue_size = len(self.priority_queue_list)
                total_available = normal_queue_size + priority_queue_size
                
                if total_available == 0:
                    consecutive_empty_cycles += 1
                    if consecutive_empty_cycles >= max_empty_cycles:
                        self.logger.warning(f"Worker {worker_name}: No requests for {max_empty_cycles * 3:.1f} seconds")
                        consecutive_empty_cycles = 0
                    continue
                
                # 从队列中取出所有请求
                for _ in range(total_available):
                    request = await self._get_next_request()
                    if request is None:
                        break
                    requests_to_process.append(request)
                
                # 如果没有请求，继续下一轮
                if not requests_to_process:
                    consecutive_empty_cycles += 1
                    if consecutive_empty_cycles >= max_empty_cycles:
                        self.logger.warning(f"Worker {worker_name}: No requests for {max_empty_cycles * 3:.1f} seconds")
                        consecutive_empty_cycles = 0
                    continue
                
                # 重置空循环计数器
                consecutive_empty_cycles = 0
                
                # 记录处理信息
                normal_queue_size = self.request_queue.qsize()
                priority_queue_size = len(self.priority_queue_list)
                total_pending = normal_queue_size + priority_queue_size
                
                self.logger.info(f"Worker {worker_name}: Processing {len(requests_to_process)} requests "
                               f"(队列总长度: {total_pending})")
                
                # 依次处理每个请求
                for request in requests_to_process:
                    # 创建异步处理任务
                    task = asyncio.create_task(
                        self._process_request_async(request, worker_name)
                    )
                    
                    self.logger.debug(f"Worker {worker_name}: Started processing request {request.request_id} "
                                   f"(priority={request.priority})")
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
                await asyncio.sleep(1.0)

        self.logger.info(f"Worker {worker_name} stopped")

    async def _process_request_async(self, request: QueuedRequest, worker_name: str):
        """异步处理单个请求"""
        try:
            self.logger.debug(f"Worker {worker_name}: Starting async processing for {request.request_id}")

            # 处理请求
            result = await self._process_request(request, worker_name)

            # 将结果发送到客户端的响应队列
            if request.client_id in self.response_queues:
                await self.response_queues[request.client_id].put(result)
                self.logger.debug(f"Worker {worker_name}: Result sent to response queue for {request.client_id}")
            else:
                self.logger.warning(f"Worker {worker_name}: No response queue found for client {request.client_id}")

        except Exception as e:
            self.logger.error(f"Error in async processing for {request.request_id}: {str(e)}")
            
            # 如果是vLLM引擎相关的错误，尝试abort请求
            if "Waiting sequence group should have only one prompt sequence" in str(e):
                self.logger.warning(f"Detected vLLM sequence group error for {request.request_id}, attempting abort")
                try:
                    # 尝试从vLLM引擎中abort这个请求
                    if hasattr(self, 'openai_client') and self.openai_client:
                        # 如果有vLLM引擎访问权限，尝试abort
                        from config.Config import GLOBAL_CONFIG
                        vllm_engine = GLOBAL_CONFIG.get('vllm_engine')
                        if vllm_engine and hasattr(vllm_engine, 'abort_request'):
                            await vllm_engine.abort_request(request.request_id)
                            self.logger.debug(f"Successfully aborted problematic request {request.request_id}")
                except Exception as abort_error:
                    self.logger.warning(f"Failed to abort problematic request {request.request_id}: {abort_error}")
            
            # 发送None作为错误结果
            if request.client_id in self.response_queues:
                try:
                    await self.response_queues[request.client_id].put(None)
                except Exception:
                    pass

    async def stop(self):
        """停止队列管理器"""
        self.logger.info("Stopping request queue manager")
        self.workers_running = False
        self.is_running = False

        # 停止队列监控协程
        if self.queue_monitor_task and not self.queue_monitor_task.done():
            self.queue_monitor_task.cancel()
            try:
                await self.queue_monitor_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Queue monitor task stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_time = time.time() - self.start_time if self.start_time else 0

        # 创建优先级队列的快照以避免竞争条件
        priority_queue_size = len(self.priority_queue_list)

        stats = {
            'total_requests_processed': self.total_requests_processed,
            'total_time': total_time,
            'requests_per_second': self.total_requests_processed / total_time if total_time > 0 else 0,
            'queue_size': self.request_queue.qsize(),
            'priority_queue_size': priority_queue_size,
            'client_stats': self.client_stats.copy(),
            'client_token_stats': self.client_token_stats.copy()
        }

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        self.logger.info("=== Queue Manager Statistics ===")
        self.logger.info(f"Total requests processed: {stats['total_requests_processed']}")
        self.logger.info(f"Requests per second: {stats['requests_per_second']:.2f}")
        self.logger.info(f"Current queue size: {stats['queue_size']}")

        for client_id, client_stats in stats['client_stats'].items():
            avg_wait = client_stats['total_wait_time'] / max(client_stats['completed_requests'], 1)
            success_rate = client_stats['completed_requests'] / max(client_stats['total_requests'], 1) * 100

            # 获取token统计信息
            token_stats = stats['client_token_stats'].get(client_id, {})
            total_input = token_stats.get('total_input_tokens', 0)
            total_output = token_stats.get('total_output_tokens', 0)
            actual_used = token_stats.get('actual_tokens_used', 0)

            self.logger.info(
                f"Client {client_id}: {client_stats['completed_requests']}/{client_stats['total_requests']} "
                f"(success: {success_rate:.1f}%, avg_wait: {avg_wait:.3f}s)")
            self.logger.info(f"  Tokens - Input: {total_input}, Output: {total_output}, "
                             f"Total Used: {actual_used}")

    async def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up queue manager resources")

        # 只停止当前实验的workers
        self.workers_running = False

        # 清空队列
        queue_cleared_count = 0
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
                queue_cleared_count += 1
            except asyncio.QueueEmpty:
                break

        # 清空优先级队列（使用锁保护）
        async with self.priority_queue_lock:
            priority_queue_cleared_count = len(self.priority_queue_list)
            self.priority_queue_list.clear()
            # 清空优先级分布缓存
            self.priority_distribution_cache.clear()

        # 记录清空的请求数量
        total_cleared = queue_cleared_count + priority_queue_cleared_count
        if total_cleared > 0:
            self.logger.info(f"Cleared {total_cleared} requests during cleanup "
                             f"(queue: {queue_cleared_count}, priority: {priority_queue_cleared_count})")

        # 重置统计信息
        self.total_requests_processed = 0
        self.start_time = None

        # 重置所有客户端的token统计
        for client_id in self.client_token_stats:
            self.client_token_stats[client_id] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'actual_tokens_used': 0
            }

        self.logger.info("Queue manager cleanup completed")

    def get_active_request_ids(self, client_id: str = None) -> List[str]:
        """获取活跃的request_id列表
        
        Args:
            client_id: 如果指定，只返回该客户端的request_id；否则返回所有
            
        Returns:
            活跃的request_id列表
        """
        active_request_ids = []

        # 检查普通队列
        temp_queue = []
        try:
            while not self.request_queue.empty():
                request = self.request_queue.get_nowait()
                if client_id is None or request.client_id == client_id:
                    active_request_ids.append(request.request_id)
                temp_queue.append(request)
        except asyncio.QueueEmpty:
            pass

        # 将请求放回队列
        for request in temp_queue:
            try:
                self.request_queue.put_nowait(request)
            except asyncio.QueueFull:
                self.logger.warning(f"Queue full when restoring request {request.request_id}")

        # 检查优先级队列（创建快照以避免竞争条件）
        # 注意：这里可能存在轻微的竞争条件，但对于统计目的是可接受的
        priority_queue_snapshot = self.priority_queue_list.copy()
        for request in priority_queue_snapshot:
            if client_id is None or request.client_id == client_id:
                active_request_ids.append(request.request_id)

        return active_request_ids

    async def abort_requests(self, request_ids: List[str]) -> int:
        """终止指定的请求
        
        Args:
            request_ids: 要终止的request_id列表
            
        Returns:
            成功终止的请求数量
        """
        if not request_ids:
            return 0

        aborted_count = 0
        request_ids_set = set(request_ids)

        # 从普通队列中移除
        temp_queue = []
        try:
            while not self.request_queue.empty():
                request = self.request_queue.get_nowait()
                if request.request_id not in request_ids_set:
                    temp_queue.append(request)
                else:
                    aborted_count += 1
                    self.logger.debug(f"Aborted request from queue: {request.request_id}")
        except asyncio.QueueEmpty:
            pass

        # 将未终止的请求放回队列
        for request in temp_queue:
            try:
                await self.request_queue.put(request)
            except asyncio.QueueFull:
                self.logger.warning(f"Queue full when restoring request {request.request_id}")

        # 从优先级队列中移除（使用锁保护）
        async with self.priority_queue_lock:
            original_priority_queue = self.priority_queue_list.copy()
            self.priority_queue_list = []

            for request in original_priority_queue:
                if request.request_id not in request_ids_set:
                    self.priority_queue_list.append(request)
                else:
                    aborted_count += 1
                    # 更新优先级分布缓存（减量更新）
                    if request.priority in self.priority_distribution_cache:
                        self.priority_distribution_cache[request.priority] -= 1
                        if self.priority_distribution_cache[request.priority] <= 0:
                            del self.priority_distribution_cache[request.priority]
                    self.logger.debug(f"Aborted request from priority queue: {request.request_id}")

        if aborted_count > 0:
            self.logger.info(f"Successfully aborted {aborted_count} requests from queue")

        return aborted_count

    def _validate_priority_cache(self) -> bool:
        """验证优先级分布缓存的一致性（用于调试）"""
        actual_distribution = {}
        for request in self.priority_queue_list:
            priority = request.priority
            actual_distribution[priority] = actual_distribution.get(priority, 0) + 1

        # 比较缓存和实际分布
        cache_keys = set(self.priority_distribution_cache.keys())
        actual_keys = set(actual_distribution.keys())

        if cache_keys != actual_keys:
            self.logger.warning(f"Priority cache key mismatch: cache={cache_keys}, actual={actual_keys}")
            return False

        for priority in cache_keys:
            if self.priority_distribution_cache[priority] != actual_distribution[priority]:
                self.logger.warning(f"Priority cache count mismatch for {priority}: "
                                    f"cache={self.priority_distribution_cache[priority]}, "
                                    f"actual={actual_distribution[priority]}")
                return False

        return True

    def _rebuild_priority_cache(self):
        """重建优先级分布缓存"""
        self.priority_distribution_cache.clear()
        for request in self.priority_queue_list:
            priority = request.priority
            self.priority_distribution_cache[priority] = self.priority_distribution_cache.get(priority, 0) + 1
        self.logger.debug(f"Priority cache rebuilt: {self.priority_distribution_cache}")
