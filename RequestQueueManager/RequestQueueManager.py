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


class QueueStrategy(Enum):
    """队列调度策略"""
    FIFO = "fifo"  # 先进先出
    PRIORITY = "priority"  # 基于优先级
    ROUND_ROBIN = "round_robin"  # 轮询
    SHORTEST_JOB_FIRST = "sjf"  # 最短作业优先
    FAIR_SHARE = "fair_share"  # 公平共享
    VTC = "vtc"


@dataclass
class  QueuedRequest:
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
        # 如果没有request_id，生成一个
        if not self.request_id:
            self.request_id = f"request_{self.client_id}_{self.worker_id}_{int(time.time() * 1000)}"
    
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
        
        # 部分优先级配置
        self.priority_insert_multiplier = 1  # 优先级倍数，优先级N可以往前插N*multiplier个位置
        self.max_priority_positions = 100  # 最大优先级插入位置限制
        
        # 统计信息
        self.total_requests_processed = 0
        self.start_time = None
        
        # 队列监控配置
        self.queue_monitor_interval = GLOBAL_CONFIG.get("queue_monitor_interval", 5)  # 队列监控间隔（秒）
        self.queue_monitor_task = None  # 队列监控任务
        
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
            fh.setLevel(logging.INFO)
            
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
    
    def configure_partial_priority(self, insert_multiplier: int = 3, max_positions: int = 20):
        """配置部分优先级参数
        
        Args:
            insert_multiplier: 优先级倍数，优先级N可以往前插N*multiplier个位置
            max_positions: 最大优先级插入位置限制
        """
        self.priority_insert_multiplier = insert_multiplier
        self.max_priority_positions = max_positions
        self.logger.info(f"Configured partial priority: multiplier={insert_multiplier}, max_positions={max_positions}")
        
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
        self.logger.info(f"Registered client: {client_id} (type: {client_type})")
    
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
                
                self.logger.info(f"[queue manager 队列监控] 总待处理: {total_pending} (普通队列: {normal_queue_size}, 优先级队列: {priority_queue_size}{priority_info})")
                
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
        
        # 如果没有提供request_id，生成一个
        if request_id is None:
            request_id = f"request_{client_id}_{worker_id}_{int(time.time() * 1000)}"
        
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
                # 修改部分优先级策略：数字越小的优先级越高，插入位置越靠前
                # 注意：负数优先级表示更高的优先级，0表示默认优先级，正数表示较低的优先级
                # 使用锁保护优先级队列操作
                async with self.priority_queue_lock:
                    # 确定插入位置
                    if len(self.priority_queue_list) == 0:
                        # 队列为空，直接添加
                        self.priority_queue_list.append(request)
                        insert_pos = 0
                    else:
                        # 遍历队列找到合适的插入位置
                        # 找到第一个优先级值大于当前请求的位置
                        insert_pos = 0
                        for i, queued_req in enumerate(self.priority_queue_list):
                            if queued_req.priority > request.priority:
                                insert_pos = i
                                break
                            else:
                                insert_pos = i + 1  # 如果没找到，插入到末尾
                        
                        # 插入到计算出的位置
                        self.priority_queue_list.insert(insert_pos, request)
                
                self.logger.info(f"Added request to priority queue: {client_id} (request_id: {request_id}, priority: {priority}, inserted at position: {insert_pos}/{len(self.priority_queue_list)})")
            else:
                # 其他策略使用普通队列
                await self.request_queue.put(request)
                self.logger.info(f"Submitted request from {client_id} to queue (request_id: {request_id})")
            
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
                return self.priority_queue_list.pop(0)  # 从头部取出（FIFO基础上的部分优先级）
            return None
    
    async def _get_round_robin_request(self) -> Optional[QueuedRequest]:
        """轮询策略"""
        # 简化的轮询实现，实际使用时可以更复杂
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
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
        if not self.openai_client:
            self.logger.error("OpenAI client not set")
            return None
        else:
            selected_client = self.openai_client[int(worker_name.split('-')[1]) % len(self.openai_client)]
        
        wait_time = time.time() - request.submit_time
        self.client_stats[request.client_id]['total_wait_time'] += wait_time
        
        try:
            # 调用原有的make_request函数，传递request_id
            result = await make_request(
                client=selected_client,
                experiment=request.experiment,
                request=request.request_content,
                start_time=request.start_time,
                request_id=request.request_id,  # 传递request_id
                priority=request.priority
            )
            
            if result is None:
                self.client_stats[request.client_id]['failed_requests'] += 1
                self.logger.debug(f"Request failed: {request.request_id}")
                return None
                
            self.client_stats[request.client_id]['completed_requests'] += 1
            self.total_requests_processed += 1
            self.logger.debug(f"Request completed: {request.request_id}")
            
            try:
                # 从result中提取token信息并更新统计
                # result格式: (output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count, slo_compliance)
                if isinstance(result, (tuple, list)) and len(result) >= 6:  # 确保有6个返回值
                    output_tokens = int(result[0])
                    input_token_count = int(result[4])
                    
                    self.client_token_stats[request.client_id]['total_output_tokens'] += output_tokens
                    self.client_token_stats[request.client_id]['total_input_tokens'] += input_token_count
                    self.client_token_stats[request.client_id]['actual_tokens_used'] += (output_tokens + input_token_count)
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
        
        # 创建工作协程
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]
        
        # 启动队列监控协程
        self.queue_monitor_task = asyncio.create_task(self._monitor_queue_status())
        
        try:
            # 等待所有工作协程和监控协程
            await asyncio.gather(*workers, self.queue_monitor_task)
        except Exception as e:
            self.logger.error(f"Error in queue processing: {e}")
        finally:
            self.is_running = False
            self.workers_running = False
    
    async def _worker(self, worker_name: str):
        """工作协程"""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.workers_running:
            try:
                request = await self._get_next_request()
                if request is None:
                    await asyncio.sleep(0.1)  # 没有请求时短暂休眠
                    continue
                
                if not isinstance(request, QueuedRequest):
                    self.logger.error(f"Invalid request type: {type(request)}")
                    continue
                
                # 处理请求
                try:
                    result = await self._process_request(request, worker_name)
                    
                    # 将结果发送到客户端的响应队列
                    if request.client_id in self.response_queues:
                        await self.response_queues[request.client_id].put(result)
                except Exception as e:
                    self.logger.error(f"Error in _process_request: {str(e)}")
                    if request.client_id in self.response_queues:
                        await self.response_queues[request.client_id].put(None)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
                await asyncio.sleep(1.0)
        
        self.logger.info(f"Worker {worker_name} stopped")
    
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
            
            self.logger.info(f"Client {client_id}: {client_stats['completed_requests']}/{client_stats['total_requests']} "
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
                    self.logger.debug(f"Aborted request from priority queue: {request.request_id}")
        
        if aborted_count > 0:
            self.logger.info(f"Successfully aborted {aborted_count} requests from queue")
        
        return aborted_count 