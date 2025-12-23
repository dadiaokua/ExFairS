import asyncio
import json
import logging
import subprocess
import os
from datetime import datetime, timedelta
import time
from functools import wraps

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_results
from util.MathUtil import fairness_result, is_fairness_LFSLLM, is_fairness_QUE, is_fairness_FCFS

def timing_decorator(func):
    """装饰器：用于测量函数执行时间"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 获取实例的timing_stats
        self = args[0]
        if not hasattr(self, 'timing_stats'):
            self.timing_stats = {}
        if func.__name__ not in self.timing_stats:
            self.timing_stats[func.__name__] = []
        self.timing_stats[func.__name__].append(execution_time)
        
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 获取实例的timing_stats
        self = args[0]
        if not hasattr(self, 'timing_stats'):
            self.timing_stats = {}
        if func.__name__ not in self.timing_stats:
            self.timing_stats[func.__name__] = []
        self.timing_stats[func.__name__].append(execution_time)
        
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class ExperimentMonitor:
    """
    实验监控器类，负责监控实验结果、计算公平性并触发资源调整
    """

    def __init__(self, clients, result_queue, client_count, exp_type, request_queue, use_tunnel, config=None):
        """
        初始化监控器

        Args:
            clients: 客户端列表
            result_queue: 结果队列
            client_count: 客户端数量
            exp_type: 实验类型 (LFS, VTC, DLPM等)
            config: 配置参数，默认使用GLOBAL_CONFIG
        """
        self.clients = clients
        self.result_queue = result_queue
        self.request_queue = request_queue
        self.client_count = client_count
        self.exp_type = exp_type
        self.config = config or GLOBAL_CONFIG
        self.tmp_results = []
        self.fairness_results = []
        self.start_time = None
        self.logger = self._setup_logger()
        self.log_gpu_data = use_tunnel
        self.timing_stats = {}  # 用于存储时间统计
        
        # 为这个监控器实例生成唯一的时间戳
        self.monitor_timestamp = datetime.now().strftime("%m_%d_%H_%M")
        
        # 设置结果文件名，包含实验类型
        self.results_file = f'tmp_result/tmp_fairness_result_{self.exp_type}_{GLOBAL_CONFIG.get("monitor_file_time")}.json'

        # 设置公平性调整策略映射
        self.fairness_strategies = {
            "LFS": is_fairness_LFSLLM,
            "ExFairS": is_fairness_LFSLLM,
            # "VTC": is_fairness_VTC,
            "QUEUE_LFS": is_fairness_LFSLLM,
            "QUEUE_ExFairS": is_fairness_LFSLLM,
            "QUEUE_MINQUE": is_fairness_QUE,
            # "QUEUE_VTC": is_fairness_VTC,
            "Justitia": is_fairness_LFSLLM,  # Justitia uses similar fairness calculation
            "SLOGreedy": is_fairness_LFSLLM,  # SLOGreedy uses similar fairness calculation
            "QUEUE_Justitia": is_fairness_LFSLLM,
            "QUEUE_SLOGreedy": is_fairness_LFSLLM,
        }

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(f"ExperimentMonitor-{self.exp_type}")
        
        # 确保日志记录器没有被重复配置
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # 创建控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 确保log目录存在
            os.makedirs('log', exist_ok=True)
            
            # 使用全局配置中的时间戳
            timestamp = GLOBAL_CONFIG.get("monitor_file_time", "default")
            
            # 创建文件处理器
            fh = logging.FileHandler(filename=f'log/monitor_{self.exp_type}_{timestamp}.log', encoding="utf-8", mode="a")
            fh.setLevel(logging.INFO)
            
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

    async def __call__(self):
        """
        使类实例可以作为协程调用
        这样可以直接将实例传递给 asyncio.create_task()
        """
        return await self.start_monitoring()

    async def start_monitoring(self):
        """开始监控实验结果"""
        self.start_time = datetime.now()
        self.logger.info(f'Starting monitor for {self.exp_type} experiment with {self.client_count} clients')
        exp_duration = timedelta(seconds=self.config['exp_time'])

        while datetime.now() - self.start_time < exp_duration:
            await self._check_results()
            if self.log_gpu_data == 0:
                self._log_gpu_status()
            await asyncio.sleep(5)  # 每5秒检查一次

        self.logger.info(f'Experiment duration reached. Monitoring stopped.')
        return self.fairness_results

    def _log_gpu_status(self):
        """记录当前GPU利用率和显存占用"""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            output = result.stdout.strip()
            log_lines = []
            for line in output.split('\n'):
                parts = [x.strip() for x in line.split(',')]
                if len(parts) != 7:
                    self.logger.warning(f"Unexpected nvidia-smi output line: {line}")
                    continue

                timestamp, index, util_gpu, util_mem, sm_clock, mem_used, mem_total = parts
                log_line = (f"[{timestamp}] GPU {index} | GPU Util: {util_gpu}% | "
                            f"Mem BW Util: {util_mem}% | SM Clock: {sm_clock} MHz | "
                            f"Memory: {mem_used}/{mem_total} MiB")
                log_lines.append(log_line)

            # 生成GPU监控日志文件路径
            gpu_log_file = f"tmp_result/gpu_monitor_log_{self.monitor_timestamp}.txt"
            
            # 确保目录存在
            os.makedirs("tmp_result", exist_ok=True)
            
            # 追加记录到文件
            with open(gpu_log_file, "a", encoding="utf-8") as f:
                for log in log_lines:
                    f.write(log + "\n")
                    
            self.logger.debug(f"GPU status logged to {gpu_log_file}")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to run nvidia-smi: {e}")
        except FileNotFoundError:
            self.logger.warning("nvidia-smi not found. GPU monitoring disabled.")
        except Exception as e:
            self.logger.warning(f"Failed to fetch GPU status: {e}")

    async def _check_request_queue(self):
        requests = []
        while not self.request_queue.empty():
            request = self.request_queue.get()
            requests.append(request)
        return requests

    async def _check_results(self):
        """检查结果队列并处理结果"""
        if not self.result_queue.empty():
            self.logger.info('Queue not empty, getting next result...')
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=10)
                self.tmp_results.append(result)
                self.logger.info(f'Current completed client numbers: {len(self.tmp_results)}')

                if len(self.tmp_results) == self.client_count:
                    await self._process_complete_round()

                self.result_queue.task_done()
                self.logger.info('Task marked as done')
            except asyncio.TimeoutError:
                self.logger.warning("Timeout while waiting for result.")

    @timing_decorator
    async def _process_complete_round(self):
        """处理完整一轮的结果"""
        try:
            # 计算公平性
            jains_indices, s_result = await self._calculate_fairness()
            
            # jains_indices is now a dict with keys: safi, token, slo_violation
            self.logger.info(f"Fairness calculation complete:")
            self.logger.info(f"  SAFI JAIN: {jains_indices['safi']:.4f}")
            self.logger.info(f"  Token JAIN: {jains_indices['token']:.4f}")
            self.logger.info(f"  SLO Violation JAIN: {jains_indices['slo_violation']:.4f}")
            self.logger.info(f"  Service result: {s_result}")
            
            # 根据配置决定是否进行公平性调整
            if "LFS" in self.exp_type or "QUE" in self.exp_type:
                exchange_count = await self._adjust_fairness()
            else:
                exchange_count = 0
                self.logger.info("Skipping fairness adjustment (FCFS)")
            
            # 保存结果
            self.logger.info(f"Saving results... {jains_indices}, {s_result}, {exchange_count}")
            self._save_results(jains_indices, s_result, exchange_count)
            self.logger.info("Results saved successfully")
            
            # 重置客户端
            await self._reset_clients()
            
            # 清空临时结果
            self.tmp_results = []

            # 打印详细的时间统计
            # self._print_timing_stats()
            
        except Exception as e:
            self.logger.error(f"Error in _process_complete_round: {str(e)}", exc_info=True)
            raise

    @timing_decorator
    async def _calculate_fairness(self):
        """计算公平性指标"""
        return await fairness_result(self.clients, self.exp_type, self.logger)

    @timing_decorator
    async def _adjust_fairness(self):
        """根据实验类型调整公平性"""
        self.logger.info(f"Starting fairness adjustment for {self.exp_type}...")

        # 从映射中获取调整函数
        adjust_function = self.fairness_strategies.get(self.exp_type)

        if adjust_function:
            exchange_count = await adjust_function(self.clients, self.exp_type)
        else:
            exchange_count = 0
            self.logger.warning(f"Invalid experiment type: {self.exp_type}, skipping fairness")

        self.logger.info("Fairness adjustment complete")
        return exchange_count

    @timing_decorator
    def _save_results(self, jains_indices, s_result, exchange_count):
        """保存结果到文件
        
        Args:
            jains_indices: Dictionary with keys 'safi', 'token', 'slo_violation'
            s_result: Service result
            exchange_count: Number of resource exchanges
        """
        try:
            self.logger.info("Starting to save results...")
            results_file = self.results_file

            # 确保结果文件存在
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            # 保存结果 - 传递完整的jains_indices字典
            save_results(exchange_count, jains_indices, s_result, results_file)
            self.fairness_results.append((jains_indices, s_result))
            
            self.logger.info(f"Results saved to {results_file}: SAFI={jains_indices['safi']:.4f}, "
                           f"Token={jains_indices['token']:.4f}, SLO={jains_indices['slo_violation']:.4f}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise

    @timing_decorator
    async def _reset_clients(self):
        """重置所有客户端"""
        self.logger.info("Notifying clients of completion...")
        for i, client in enumerate(self.clients):
            self.logger.info(f"Resetting client {i + 1}/{len(self.clients)}")
            client.exchange_Resources_Times = 0
            client.monitor_done_event.set()
        self.logger.info("All clients notified")

    def _print_timing_stats(self):
        """打印时间统计信息并保存到文件"""
        # 创建time_log目录
        if not os.path.exists('time_log'):
            os.makedirs('time_log')

        # 生成时间戳作为文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'time_log/timing_stats_{timestamp}.txt'

        # 准备输出内容
        output = ["\n=== 时间统计 ==="]
        for operation, times in self.timing_stats.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            stats = [
                f"{operation}:",
                f"  平均时间: {avg_time:.4f} 秒",
                f"  最长时间: {max_time:.4f} 秒", 
                f"  最短时间: {min_time:.4f} 秒",
                f"  执行次数: {len(times)}"
            ]
            output.extend(stats)
        output.append("==============\n")

        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))

        # 同时写入日志
        self.logger.info('\n'.join(output))
