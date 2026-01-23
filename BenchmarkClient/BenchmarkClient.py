import asyncio
import logging
import math
import os
import random

from config.Config import GLOBAL_CONFIG
from experiment.queue_experiment import QueueExperiment
from RequestQueueManager.RequestQueueManager import QueueStrategy


class BenchmarkClient:
    """Class representing a benchmark client with its configurations and state"""

    def __init__(self, client_type, client_index, qpm, port, api_key, tokenizer, exp_type,
                 distribution, request_timeout, concurrency, round, round_time, sleep, time_data,
                 result_queue, formatted_json, OpenAI_client, qpm_ratio, latency_slo, use_time_data=0, 
                 queue_manager=None, qpm_variation=0.0, qpm_pattern="random", 
                 burst_interval=3, burst_multiplier=2.0):
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
            qpm_variation: QPM 波动范围 (0-1)
            qpm_pattern: QPM 变化模式 ("random", "burst", "ramp", "wave")
            burst_interval: 突发间隔（仅 burst 模式）
            burst_multiplier: 突发倍数（仅 burst 模式）
        """
        self.client_type = client_type
        self.client_index = client_index
        self.client_id = f"{client_type}_{client_index}"
        self.base_qpm = qpm  # 保存基础 QPM
        self.qpm = qpm
        self.qpm_ratio = qpm_ratio
        
        # QPM 动态变化配置
        self.qpm_variation = qpm_variation
        self.qpm_pattern = qpm_pattern
        self.burst_interval = burst_interval
        self.burst_multiplier = burst_multiplier
        
        # 预生成的 QPM 因子序列（用于跨策略一致性）
        self.qpm_factor_sequence: list = None  # 由外部注入
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
        self.que = 0
        self.credit = 0
        self.max_service = -1
        self.priority = 0

        self.openAI_client = OpenAI_client
        self.monitor_done_event = asyncio.Event()

        # QuE
        self.Norm_throughput = 0.0
        self.Norm_latency = 0.0
        self.Norm_cost = 0.0

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
            # 文件处理器 - 记录所有详细信息
            fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # 控制台输出 - 只显示警告和错误
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
        # 防止日志传播到父logger，避免其他logger配置的干扰
        logger.propagate = False

        return logger

    def calculate_dynamic_qpm(self, round_index: int) -> float:
        """
        根据轮次和配置计算动态 QPM
        
        Args:
            round_index: 当前轮次索引 (0-based)
            
        Returns:
            计算后的 QPM 值
        """
        if self.qpm_variation <= 0:
            # 无波动，返回基础 QPM
            return self.base_qpm
        
        base = self.base_qpm
        variation = self.qpm_variation
        
        if self.qpm_pattern == "random":
            # 随机波动模式
            factor = 1.0
            
            # 【关键】优先使用预生成的 QPM 因子序列（确保跨策略一致性）
            if self.qpm_factor_sequence is not None and round_index < len(self.qpm_factor_sequence):
                factor = self.qpm_factor_sequence[round_index]
                self.logger.debug(f"Client {self.client_id}: Using pre-generated QPM factor={factor:.3f} for round {round_index}")
            elif hasattr(self, 'realtime_monitor') and self.realtime_monitor:
                # 从监控器获取协调后的 QPM 因子（向后兼容）
                factor = self.realtime_monitor.get_client_qpm_factor(self.client_id)
                self.logger.debug(f"Client {self.client_id}: Using coordinated QPM factor={factor:.3f}")
            else:
                # 没有监控器也没有预生成序列，独立计算（向后兼容）
                factor = 1 + random.uniform(-variation, variation)
            dynamic_qpm = base * factor
            
        elif self.qpm_pattern == "burst":
            # 突发模式：每隔 burst_interval 轮有一次高负载
            if round_index % self.burst_interval == 0:
                dynamic_qpm = base * self.burst_multiplier
            else:
                # 非突发轮次，正常随机波动（较小范围）
                factor = 1 + random.uniform(-variation * 0.5, variation * 0.5)
                dynamic_qpm = base * factor
                
        elif self.qpm_pattern == "ramp":
            # 渐增模式：从 (1-variation)*base 逐渐增加到 (1+variation)*base
            total_rounds = self.round
            if total_rounds > 1:
                progress = round_index / (total_rounds - 1)
            else:
                progress = 0.5
            factor = (1 - variation) + 2 * variation * progress
            dynamic_qpm = base * factor
            
        elif self.qpm_pattern == "wave":
            # 正弦波模式：QPM 在周期内波动
            total_rounds = self.round
            # 一个完整周期
            phase = 2 * math.pi * round_index / max(total_rounds, 1)
            factor = 1 + variation * math.sin(phase)
            dynamic_qpm = base * factor
            
        else:
            # 未知模式，返回基础 QPM
            dynamic_qpm = base
        
        # 确保 QPM 不低于 1
        dynamic_qpm = max(1, dynamic_qpm)
        
        self.logger.info(f"Client {self.client_id}: Round {round_index + 1} dynamic QPM: "
                        f"{dynamic_qpm:.1f} (base={base}, pattern={self.qpm_pattern}, variation={variation})")
        
        return dynamic_qpm

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
            # 计算动态 QPM
            self.qpm = self.calculate_dynamic_qpm(i)
            # 应用 QPM ratio（如果需要）
            self.qpm = self.qpm * self.qpm_ratio
            print(f"Client {self.client_id}: Running configuration {i + 1}/{self.round}: QPM={self.qpm:.1f}")
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

        # 实验类型映射（只支持队列模式）
        experiment_types = {
            "QUEUE_FCFS": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.FIFO),
            "QUEUE_LFS": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.PRIORITY),
            "QUEUE_ExFairS": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.PRIORITY),
            "QUEUE_RR": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.ROUND_ROBIN),
            "QUEUE_ROUND_ROBIN": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.ROUND_ROBIN),
            "QUEUE_VTC": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.VTC),
            "QUEUE_Justitia": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.JUSTITIA),
            "QUEUE_SLOGreedy": lambda client: QueueExperiment(client, self.queue_manager, QueueStrategy.SLO_GREEDY),
        }

        # 创建并运行实验
        experiment_creator = experiment_types.get(self.exp_type)
        if experiment_creator is None:
            raise ValueError(f"Unknown experiment type: {self.exp_type}. "
                           f"Available types: {list(experiment_types.keys())}")
        
        self.experiment = experiment_creator(self)
        await self.experiment.setup()
        result = await self.experiment.run(config_round)

        return result, self.experiment

    def start(self):
        """Start the benchmark task"""
        self.task = asyncio.create_task(self.run_all_benchmarks())
        return self.task
