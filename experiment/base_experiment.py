import asyncio
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import logging

from util.JsonFormatterUtil import make_prefix_list
from util.MathUtil import calculate_metrics
from util.RequestUtil import worker


class BaseExperiment:
    """
    表示单个基准测试实验的类
    负责设置、运行和收集实验结果
    """

    def __init__(self, client):
        """
        初始化基准测试实验

        Args:
            client: 运行实验的客户端实例
            config: 实验配置参数
        """
        self.client = client
        self.config = client.experiment_config
        self.logger = client.logger
        self.priority = client.priority

        # 从客户端获取必要的属性
        self.client_id = client.client_id
        self.openAI_client = client.openAI_client
        self.concurrency = client.concurrency
        self.tokenizer = client.tokenizer
        self.request_timeout = client.request_timeout
        self.round_time = client.round_time
        self.distribution = client.distribution
        self.formatted_json = client.formatted_json
        self.time_data = client.time_data
        self.use_time_data = client.use_time_data

        # 实验特定参数
        self.output_tokens = self.config.get('output_tokens', 200)
        self.qpm = self.config.get('qpm', 1)
        self.config_round = self.config.get('config_round', 1)
        self.latency_slo = self.config.get('latency_slo', 10)
        
        # 添加SLO调试日志
        self.logger.info(f"Client {self.client_id}: Experiment initialized with SLO = {self.latency_slo}s (seconds)")
        
        self.active_ratio = client.active_ratio
        self.time_ratio = client.time_ratio
        self.max_service = client.max_service

        self.qps_per_worker = 0
        self.exp_type = ''
        self.drift_time = 0
        self.timeout_count = 0

        # 实验结果
        self.experiment_results = []
        self.start_time = None
        self.end_time = None
        self.num_requests = 0
        self.metrics = None

    async def setup(self):
        """设置实验，进行必要的准备工作"""
        assert self.openAI_client is not None, "OpenAI Client must not be None"

        print(
            f"[Client {self.client_id}] Experiment setup complete: {self.qpm} workers with {1} QPS per worker")
        return self

    async def run(self, config_round):
        """运行实验并收集结果"""

        self.logger.info(
            f"Starting benchmark round {config_round} run with QPS={self.qpm}, output_tokens={self.output_tokens}")
        self.logger.info(f"Client ID: {self.client_id}, Concurrency: {self.concurrency}")
        self.logger.info(
            f"Time ratio: {self.time_ratio}, Active ratio: {self.active_ratio}, QPS ratio: {self.client.qpm_ratio}")

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)
        self.experiment_results = []
        workers = []

        # 记录开始时间
        self.start_time = time.time()
        self.logger.info(
            f"Benchmark started at {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        # 根据并发数创建workers
        if self.qpm < self.concurrency:
            worker_counts = 1
        else:
            worker_counts = self.concurrency
        qpm_per_worker = self.qpm / worker_counts
        self.logger.info(f"Creating {worker_counts} workers, each handling {qpm_per_worker} QPS")
        self.logger.info(f"Total formatted JSON size: {len(self.formatted_json)}")

        # Split formatted_json into equal chunks based on concurrency
        chunk_size = len(self.formatted_json) // worker_counts
        remaining = len(self.formatted_json) % worker_counts
        total_size = len(self.formatted_json)
        self.logger.info(f"Chunk size: {chunk_size}, Remaining: {remaining}")

        for worker_id in range(worker_counts):
            # Randomly select a starting point
            start_idx = random.randint(0, total_size - chunk_size - (1 if remaining > 0 else 0))
            end_idx = start_idx + chunk_size + (1 if remaining > 0 else 0)
            remaining = max(0, remaining - 1)  # Decrement remaining if needed

            self.logger.debug(f"Worker {worker_id}: start_idx={start_idx}, end_idx={end_idx}")

            # Handle wrap-around case if end_idx exceeds list length
            if end_idx > total_size:
                worker_json = self.formatted_json[start_idx:] + self.formatted_json[:end_idx - total_size]
                self.logger.debug(f"Worker {worker_id}: wrap-around case, json size={len(worker_json)}")
            else:
                worker_json = self.formatted_json[start_idx:end_idx]
                self.logger.debug(f"Worker {worker_id}: normal case, json size={len(worker_json)}")
            # if self.exp_type == "DLPM":
            #     worker_json = make_prefix_list(worker_json, self.tokenizer, 200 if "short" in self.client_id else 1000)
            # else:
            #     random.shuffle(worker_json)

            if worker_json is None:
                raise ValueError(f"sample_content is None! client_index={self.client_id}, worker_id={worker_id}")
            if not isinstance(worker_json, list):
                raise TypeError(f"sample_content is not a list! type={type(worker_json)}")
            if len(worker_json) == 0:
                raise ValueError(f"sample_content is empty! client_index={self.client_id}, worker_id={worker_id}")

            random.shuffle(worker_json)

            worker_task = asyncio.create_task(
                worker(
                    self,
                    self.openAI_client,
                    semaphore,
                    self.experiment_results,
                    worker_id,
                    worker_json,
                    qpm_per_worker
                )
            )
            workers.append(worker_task)

        self.logger.info("Waiting for all workers to complete")
        worker_results = await asyncio.gather(*workers)
        completed_requests, drift_time, total_requests = zip(*worker_results)

        # 汇总结果
        self.num_requests = sum(total_requests)
        self.drift_time = sum(drift_time)

        # 记录结束时间
        self.end_time = time.time()
        self.logger.info(
            f"Benchmark ended at {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}, total time: {self.end_time - self.start_time:.2f}s")

        # 计算指标
        return await self.calculate_results(sum(completed_requests), sum(total_requests))

    async def calculate_results(self, completed_requests, total_requests):
        """计算实验结果指标"""
        if not self.experiment_results or self.start_time is None or self.end_time is None:
            # 获取客户端ID用于日志
            client_id = getattr(self.client, 'client_id', 'unknown_client')
            self.logger.warning(f"Client {client_id}: Experiment has no results or missing timestamps - creating empty result")
            
            # 创建一个基础的失败结果，而不是返回None
            empty_metrics = {
                'successful_requests': completed_requests,
                'total_requests': total_requests,
                'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'average': 0},
                'time_to_first_token': {'p50': 0, 'p95': 0, 'p99': 0, 'average': 0},
                'tokens_per_second': {'p50': 0, 'p95': 0, 'p99': 0, 'average': 0},
                'requests_per_second': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'slo_violations': 0,
                'slo_violation_count': 0,  # 添加slo_violation_count字段
                'time': self.end_time if self.end_time else 0,  # 添加time字段
                'qps': self.qpm,  # 添加qps字段
                'concurrency': self.concurrency,  # 添加concurrency字段
                'fairness_ratio': getattr(self.client, 'fairness_ratio', 0),  # 添加fairness_ratio字段
                'credit': getattr(self.client, 'credit', 0),  # 添加credit字段
                'client_index': self.client_id,  # 添加client_index字段以解决公平性计算错误
                'experiment_failed': True,  # 标记实验失败
                'failure_reason': 'No experiment results or missing timestamps'
            }
            
            # 保存到metrics并返回
            self.metrics = empty_metrics
            return self.metrics

        self.metrics = calculate_metrics(
            self.concurrency,
            self.request_timeout,
            self.client_id,
            self.experiment_results,
            self.start_time,
            self.end_time,
            self.num_requests,
            self.qpm,
            self.output_tokens,
            self.latency_slo,
            self.client.fairness_ratio,
            self.drift_time,
            self.client.credit,
            self.timeout_count
        )

        # 更新客户端的相关指标
        self.client.avg_latency_div_standard_latency = self.metrics['avg_latency_div_standard_latency']
        self.client.slo_violation_count = self.metrics['slo_violation_count']

        # 如果成功率小于80%，则将QPS增加率设置为1
        if completed_requests / total_requests < 0.8:
            self.client.qpm_ratio = 1

        return self.metrics

    def get_summary(self):
        """获取实验结果摘要"""
        if not self.metrics:
            return {"status": "not_run", "client_id": self.client_id}

        return {
            "client_id": self.client_id,
            "qps": self.qpm,
            "output_tokens": self.output_tokens,
            "latency_slo": self.latency_slo,
            "total_requests": self.num_requests,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "avg_latency_ratio": self.metrics['avg_latency_div_standard_latency'],
            "slo_violations": self.metrics['slo_violation_count'],
            "success_rate": self.metrics['successful_requests'] / self.metrics['total_requests'] if self.metrics[
                                                                                                        'total_requests'] > 0 else 0
        }

    async def cleanup(self):
        return None
