import asyncio
import logging
import os

from config.Config import GLOBAL_CONFIG
from experiment.base_experiment import BaseExperiment

from experiment.FCFS_experiment import FCFSExperiment
from experiment.LFS_experiment import LFSExperiment
from experiment.VTC_experiment import VTCExperiment
from experiment.queue_experiment import QueueExperiment
from RequestQueueManager.RequestQueueManager import QueueStrategy


class BenchmarkClient:
    """Class representing a benchmark client with its configurations and state"""

    def __init__(self, client_type, client_index, qpm, port, api_key, tokenizer, exp_type,
                 distribution, request_timeout, concurrency, round, round_time, sleep, time_data,
                 result_queue, formatted_json, OpenAI_client, qpm_ratio, latency_slo, use_time_data=0, 
                 queue_manager=None):
        """Initialize a benchmark client

        Args:
            client_type (str): Type of client ('short' or 'long')
            client_index (int): Index of this client
            configurations (list): List of benchmark configurations
            port (list): List of local ports
            api_key (str): API key for vLLM server
            distribution (str): Distribution of requests
            request_timeout (int): Timeout for each request in seconds
            concurrency (int): Number of concurrent requests
            round_time (int): Timeout for every round in seconds
            sleep (int): Sleep time between rounds
            result_queue (asyncio.Queue): Queue for sending results
            update_event (asyncio.Event): Event for notifying monitor
            use_time_data (int): Whether to use time data
            formatted_json (list): Formatted input JSON data
            queue_manager: 共享的队列管理器实例
        """
        self.client_type = client_type
        self.client_index = client_index
        self.client_id = f"{client_type}_{client_index}"
        self.qpm = qpm
        self.qpm_ratio = qpm_ratio
        self.port = port
        self.api_key = api_key
        self.distribution = distribution
        self.request_timeout = request_timeout
        self.concurrency = concurrency
        self.round_time = round_time
        self.sleep = sleep
        self.result_queue = result_queue
        self.use_time_data = use_time_data
        self.formatted_json = formatted_json
        self.tokenizer = tokenizer
        self.time_data = time_data
        self.round = round
        self.exp_type = exp_type
        self.latency_slo = latency_slo
        self.queue_manager = queue_manager  # 添加队列管理器

        self.avg_latency_div_standard_latency = -1
        self.slo_violation_count = -1
        self.service = -1
        self.service_div_latency = -1
        self.exchange_Resources_Times = 0
        self.active_ratio = 1.0
        self.time_ratio = 1.0
        self.fairness_ratio = 0
        self.credit = 0
        self.max_service = -1
        self.priority = 0

        self.openAI_client = OpenAI_client
        self.monitor_done_event = asyncio.Event()

        # State tracking
        self.results = []
        self.task = None

        self.experiment_config = None
        self.experiment = None
        
        # 添加request ID跟踪
        self.active_request_ids = set()  # 跟踪当前活跃的请求ID

        # 设置logger（只设置一次，防止重复handler）
        self.logger = self._setup_logger()

    def _setup_logger(self):
        # 日志文件夹和文件名
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用全局配置中的时间戳
        timestamp = GLOBAL_CONFIG.get("monitor_file_time", "default")
        
        # 在文件名中加入实验类型
        log_file = os.path.join(log_dir, f"client_{self.client_id.split('_')[0]}_{self.exp_type}_{timestamp}.log")

        logger = logging.getLogger(f"client_{self.client_id}")
        logger.setLevel(logging.DEBUG)  # 将logger的级别设置为DEBUG
        if not logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")  # 改回追加模式
            fh.setLevel(logging.DEBUG)  # 将文件处理器的级别也设置为DEBUG
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # 控制台输出
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)  # 将控制台处理器的级别设置为DEBUG
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    def register_request_id(self, request_id):
        """注册一个新的请求ID"""
        self.active_request_ids.add(request_id)
        self.logger.debug(f"Client {self.client_id}: 注册请求 {request_id}")

    def unregister_request_id(self, request_id):
        """注销一个请求ID"""
        self.active_request_ids.discard(request_id)
        self.logger.debug(f"Client {self.client_id}: 注销请求 {request_id}")

    async def _abort_all_engine_requests(self):
        """终止引擎内的所有活跃请求，确保每轮测试之间的干净状态"""
        # 检查是否有直接的vLLM引擎访问
        if 'vllm_engine' not in GLOBAL_CONFIG or GLOBAL_CONFIG['vllm_engine'] is None:
            self.logger.debug(f"Client {self.client_id}: 没有vLLM引擎访问，跳过abort")
            return False
        
        # 从task_status中获取未完成的request_id
        active_request_ids = self._get_active_request_ids_from_tasks()
        
        # 如果有队列管理器，也从队列中获取活跃的request_id
        if self.queue_manager:
            queue_request_ids = self.queue_manager.get_active_request_ids(self.client_id)
            active_request_ids.update(queue_request_ids)
            self.logger.debug(f"Client {self.client_id}: 从队列管理器找到 {len(queue_request_ids)} 个队列中的请求")

        if not active_request_ids:
            self.logger.debug(f"Client {self.client_id}: 没有活跃的请求需要abort")
            return True
        
        try:
            engine = GLOBAL_CONFIG['vllm_engine']
            
            # 如果有队列管理器，先从队列中abort请求
            queue_aborted_count = 0
            if self.queue_manager:
                queue_request_ids = [rid for rid in active_request_ids if rid.startswith("request_")]
                if queue_request_ids:
                    queue_aborted_count = await self.queue_manager.abort_requests(queue_request_ids)
                    self.logger.debug(f"Client {self.client_id}: 从队列中abort了 {queue_aborted_count} 个请求")
            
            # 使用从task_status获取的request ID进行引擎abort
            engine_aborted_count = await self._abort_tracked_requests(engine, active_request_ids)
            
            total_aborted = queue_aborted_count + engine_aborted_count
            if total_aborted > 0:
                self.logger.info(f"✓ Client {self.client_id}: 已abort {total_aborted} 个请求 (队列: {queue_aborted_count}, 引擎: {engine_aborted_count})")
                # 给引擎一点时间来处理abort
                await asyncio.sleep(0.1)
                return True
            else:
                self.logger.debug(f"Client {self.client_id}: 没有成功abort任何请求")
                return False
                
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: _abort_all_engine_requests 异常: {e}")
            return False

    def _get_active_request_ids_from_tasks(self):
        """从active_request_ids中获取未完成请求的ID"""
        active_request_ids = set()
        
        # 直接使用active_request_ids机制
        if self.active_request_ids:
            active_request_ids = self.active_request_ids.copy()
            self.logger.debug(f"Client {self.client_id}: 找到 {len(active_request_ids)} 个活跃请求")
        
        return active_request_ids

    async def _abort_tracked_requests(self, engine, request_ids_to_abort):
        """使用提供的request ID列表进行abort"""
        aborted_count = 0
        failed_requests = set()

        for request_id in request_ids_to_abort:
            try:
                # 尝试abort
                success = False
                
                # 方法1: 直接调用engine.abort (异步)
                if hasattr(engine, 'abort'):
                    try:
                        await engine.abort(request_id)
                        success = True
                        self.logger.debug(f"Client {self.client_id}: 使用engine.abort成功abort {request_id}")
                    except Exception as e:
                        self.logger.debug(f"Client {self.client_id}: engine.abort失败 {request_id}: {e}")
                
                if success:
                    aborted_count += 1
                    # 从传统的active_request_ids中移除（如果存在）
                    self.active_request_ids.discard(request_id)
                else:
                    failed_requests.add(request_id)
                    
            except Exception as e:
                self.logger.debug(f"Client {self.client_id}: abort请求 {request_id} 时出现异常: {e}")
                failed_requests.add(request_id)
        
        # 记录失败的请求
        if failed_requests:
            self.logger.warning(f"Client {self.client_id}: 以下请求abort失败: {failed_requests}")
        
        return aborted_count

    async def run_all_benchmarks(self):
        """Run all benchmark configurations for this client"""
        print(f"Starting benchmarks for client {self.client_id} with {self.round} configurations")

        for i in range(self.round):
            # Run benchmark with current configuration
            self.qpm = self.qpm * self.qpm_ratio
            print(f"Client {self.client_id}: Running configuration {i + 1}/{self.round}: {self.qpm}")
            result, benchmark_experiment = await self.run_benchmark(GLOBAL_CONFIG["output_tokens"], self.qpm, i, self.latency_slo)

            # Store result first
            if result:
                self.results.append(result)
            else:
                self.logger.info(f"Client {self.client_id}: No result for configuration {i + 1}/{self.round}")

            if i != 0:
                # 等待 monitor 通知处理完成
                await self.monitor_done_event.wait()
                self.monitor_done_event.clear()

            # 现在可以安全地访问self.results[-1]，因为result已经被添加
            if self.results:  # 额外的安全检查
                self.results[-1]["fairness_ratio"] = self.fairness_ratio

            # 清理实验资源
            if benchmark_experiment and hasattr(benchmark_experiment, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(benchmark_experiment.cleanup):
                        await benchmark_experiment.cleanup()
                    else:
                        benchmark_experiment.cleanup()
                    self.logger.debug(f"Client {self.client_id}: 实验清理完成")
                except Exception as e:
                    self.logger.warning(f"Client {self.client_id}: 实验清理时出现警告: {e}")
            else:
                self.logger.debug(f"Client {self.client_id}: 实验对象无cleanup方法，跳过清理")

            # # 每次benchmark结束后，终止引擎内的所有活跃请求
            await self._abort_all_engine_requests()
            
            await self.result_queue.put(1)

            # Wait between runs
            await asyncio.sleep(self.sleep)

        return self.results

    async def run_benchmark(self, output_tokens, qpm, config_round, latency_slo):
        """
        运行基准测试实验

        Args:
            output_tokens: 每个请求的输出令牌数
            qpm: 每秒查询数
            config_round: 配置轮次
            latency_slo: 延迟服务水平目标

        Returns:
            dict: 实验结果指标
        """

        self.experiment_config = {
            'output_tokens': output_tokens,
            'qpm': qpm,
            'config_round': config_round,
            'latency_slo': latency_slo
        }

        experiment_types = {
            "baseline": BaseExperiment,
            "LFS": LFSExperiment,
            "VTC": VTCExperiment,
            "FCFS": FCFSExperiment,
            "QUEUE_FCFS": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.FIFO),
            "QUEUE_LFS": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.PRIORITY),
            "QUEUE_ROUND_ROBIN": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.ROUND_ROBIN),
            "QUEUE_SJF": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.SHORTEST_JOB_FIRST),
            "QUEUE_FAIR": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.FAIR_SHARE),
            "QUEUE_VTC": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.VTC),
        }

        # 创建并运行实验
        experiment_creator = experiment_types.get(self.exp_type, BaseExperiment)
        if callable(experiment_creator) and self.exp_type.startswith("QUEUE_"):
            self.experiment = experiment_creator(self)
        else:
            self.experiment = experiment_creator(self)
        await self.experiment.setup()
        result = await self.experiment.run(config_round)

        return result, self.experiment

    def start(self):
        """Start the benchmark task"""
        self.task = asyncio.create_task(self.run_all_benchmarks())
        return self.task
