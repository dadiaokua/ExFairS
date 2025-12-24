#!/usr/bin/env python3
"""
任务管理模块
处理基准测试任务的设置、创建和管理
"""

import asyncio
import json
import logging
import sys
import os
import random
from transformers import AutoTokenizer
from argument_parser import safe_float_conversion

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入基准测试相关模块
from BenchmarkClient.BenchmarkClient import BenchmarkClient
from util.BaseUtil import initialize_clients
from BenchmarkMonitor.BenchmarkMonitor import ExperimentMonitor
from config.Config import GLOBAL_CONFIG

# 尝试导入队列管理器
try:
    from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy
    queue_manager_available = True
except ImportError:
    queue_manager_available = False

logger = logging.getLogger(__name__)


async def setup_benchmark_tasks(args, all_results, request_queue, logger):
    """Setup and create benchmark tasks"""
    
    if not queue_manager_available:
        logger.warning("RequestQueueManager not available, queue experiments will be skipped")
    
    tasks = []
    clients = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # 加载原始prompt数据集
    logger.info("📂 加载prompt数据集...")
    
    # 使用原始prompt数据集（固定输出长度）
    short_prompts_file = "prompt_hub/short_prompts.json"
    long_prompts_file = "prompt_hub/long_prompts.json"
    
    # 检查数据集是否存在
    if not os.path.exists(short_prompts_file) or not os.path.exists(long_prompts_file):
        logger.error(f"❌ Prompt数据集不存在")
        logger.info(f"   需要的文件:")
        logger.info(f"   - {short_prompts_file}")
        logger.info(f"   - {long_prompts_file}")
        logger.info(f"   请确保prompt_hub目录下有这两个文件")
        raise FileNotFoundError(f"Required prompt files not found: {short_prompts_file}, {long_prompts_file}")
    
    # 加载prompt数据集
    with open(short_prompts_file, "r", encoding="utf-8") as f:
        short_formatted_json = json.load(f)

    with open(long_prompts_file, "r", encoding="utf-8") as f:
        long_formatted_json = json.load(f)

    mix_formatted_json = short_formatted_json + long_formatted_json
    
    logger.info(f"✅ 数据集加载完成:")
    logger.info(f"   短prompt: {len(short_formatted_json)} 条")
    logger.info(f"   长prompt: {len(long_formatted_json)} 条")
    logger.info(f"   总计: {len(mix_formatted_json)} 条")
    
    random.shuffle(mix_formatted_json)

    openAI_client = initialize_clients(args.local_port)

    # 创建共享的队列管理器（如果使用队列实验）
    queue_manager = None
    queue_manager_task = None
    if args.exp.startswith("QUEUE_") and queue_manager_available:
        # 根据实验类型选择队列策略
        strategy_map = {
            "QUEUE_FIFO": QueueStrategy.FIFO,
            "QUEUE_FCFS": QueueStrategy.FIFO,
            "QUEUE_LFS": QueueStrategy.PRIORITY,
            "QUEUE_ExFairS": QueueStrategy.PRIORITY,  # ExFairS uses priority-based scheduling
            "QUEUE_ROUND_ROBIN": QueueStrategy.ROUND_ROBIN,
            "QUEUE_VTC": QueueStrategy.VTC,
            "QUEUE_MINQUE": QueueStrategy.PRIORITY,
            "QUEUE_Justitia": QueueStrategy.JUSTITIA,  # Justitia virtual time scheduling
            "QUEUE_SLOGreedy": QueueStrategy.SLO_GREEDY  # SLO violation rate greedy scheduling
        }

        strategy = strategy_map.get(args.exp, QueueStrategy.FIFO)
        logger.info(f"Using queue strategy: {strategy.value}")
        queue_manager = RequestQueueManager(strategy=strategy, max_queue_size=20000)
        
        # 配置队列管理器
        if strategy == QueueStrategy.PRIORITY:
            # 配置部分优先级参数
            queue_manager.configure_partial_priority(
                insert_multiplier=2, 
                max_positions=50,
                delay_enabled=False,  # 禁用延迟
                max_delay=0  # 设置最大延迟为0
            )
        
        # 设置OpenAI客户端（可选，作为备用）
        if openAI_client:
            queue_manager.set_openai_client(openAI_client)
            logger.info(f"✓ OpenAI client set as fallback: {len(openAI_client)} clients")
        else:
            logger.info("No OpenAI client provided (will use vLLM engine directly)")
        
        # 验证处理能力
        vllm_engine = GLOBAL_CONFIG.get('vllm_engine')
        
        if vllm_engine is not None:
            logger.info("✓ vLLM engine is available for primary request processing")
        elif queue_manager.openai_client is not None:
            logger.info("✓ Will use OpenAI client for request processing (fallback mode)")
        else:
            logger.error("CRITICAL: Neither vLLM engine nor OpenAI client is available")
            raise RuntimeError("No request processing method available")
        
        # 动态计算 worker 数量：基于总客户端数和总 QPM
        total_clients = args.short_clients + args.long_clients + args.mix_clients
        
        # 计算总 QPM
        total_qpm = 0
        for i in range(args.short_clients):
            qpm = safe_float_conversion(args.short_qpm[0] if len(args.short_qpm) == 1 else args.short_qpm[i])
            total_qpm += qpm
        for i in range(args.long_clients):
            qpm = safe_float_conversion(args.long_qpm[0] if len(args.long_qpm) == 1 else args.long_qpm[i])
            total_qpm += qpm
        for i in range(args.mix_clients):
            qpm = safe_float_conversion(args.mix_qpm[0] if len(args.mix_qpm) == 1 else args.mix_qpm[i])
            total_qpm += qpm
        
        # 计算 worker 数量：
        # - 基础：每个客户端至少 1 个 worker
        # - 按 QPM 增加：每 10 QPM 增加 1 个 worker
        # - 最小 5，最大 50
        base_workers = total_clients
        qpm_workers = int(total_qpm / 10)
        num_workers = max(5, min(50, base_workers + qpm_workers))
        
        logger.info(f"📊 Dynamic worker calculation: {total_clients} clients, {total_qpm:.0f} total QPM → {num_workers} workers")
        
        # 启动队列管理器（在后台运行）
        queue_manager_task = asyncio.create_task(queue_manager.start_processing(num_workers=num_workers))
        logger.info(f"Created queue manager with strategy: {strategy.value}, workers: {num_workers}")
        
        # 等待一小段时间确保队列管理器正常启动
        await asyncio.sleep(2.0)
        
        # 检查队列管理器状态
        if queue_manager.is_running and queue_manager.workers_running:
            logger.info(f"✓ Queue manager started successfully: is_running={queue_manager.is_running}, workers_running={queue_manager.workers_running}")
        else:
            logger.error(f"❌ Queue manager failed to start properly. is_running: {queue_manager.is_running}, workers_running: {queue_manager.workers_running}")
            # 即使启动失败也继续，可能在后续使用中恢复

    # Create short request clients
    for index in range(args.short_clients):
        qpm_value = safe_float_conversion(args.short_qpm[0] if len(args.short_qpm) == 1 else args.short_qpm[index])
        slo_value = safe_float_conversion(
            args.short_clients_slo[0] if len(args.short_clients_slo) == 1 else args.short_clients_slo[index], 10)

        client = BenchmarkClient(
            client_type='short',
            client_index=index,
            qpm=qpm_value,
            port=args.local_port,
            api_key=args.api_key,
            distribution=args.distribution,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            round_time=args.round_time,
            sleep=args.sleep,
            result_queue=all_results,
            use_time_data=args.use_time_data,
            formatted_json=short_formatted_json,
            OpenAI_client=openAI_client,
            tokenizer=tokenizer,
            time_data=None,
            round=args.round,
            exp_type=args.exp,
            qpm_ratio=args.short_client_qpm_ratio,
            latency_slo=int(slo_value),
            queue_manager=queue_manager  # 传递队列管理器
        )
        clients.append(client)
        tasks.append(client.start())

    # Create long request clients
    for index in range(args.long_clients):
        qpm_value = safe_float_conversion(args.long_qpm[0] if len(args.long_qpm) == 1 else args.long_qpm[index])
        slo_value = safe_float_conversion(
            args.long_clients_slo[0] if len(args.long_clients_slo) == 1 else args.long_clients_slo[index], 10)

        client = BenchmarkClient(
            client_type='long',
            client_index=index,
            qpm=qpm_value,
            port=args.local_port,
            api_key=args.api_key,
            distribution=args.distribution,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            round_time=args.round_time,
            sleep=args.sleep,
            result_queue=all_results,
            use_time_data=args.use_time_data,
            formatted_json=long_formatted_json,
            OpenAI_client=openAI_client,
            tokenizer=tokenizer,
            time_data=None,
            round=args.round,
            exp_type=args.exp,
            qpm_ratio=args.long_client_qpm_ratio,
            latency_slo=int(slo_value),
            queue_manager=queue_manager  # 传递队列管理器
        )
        clients.append(client)
        tasks.append(client.start())
        
    for index in range(args.mix_clients):
        qpm_value = safe_float_conversion(args.mix_qpm[0] if len(args.mix_qpm) == 1 else args.mix_qpm[index])
        slo_value = safe_float_conversion(
            args.mix_clients_slo[0] if len(args.mix_clients_slo) == 1 else args.mix_clients_slo[index], 10)
        
        client = BenchmarkClient(
            client_type='mix',
            client_index=index,
            qpm=qpm_value,
            port=args.local_port,
            api_key=args.api_key,
            distribution=args.distribution,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            round_time=args.round_time,
            sleep=args.sleep,
            result_queue=all_results,
            use_time_data=args.use_time_data,
            formatted_json=mix_formatted_json,
            OpenAI_client=openAI_client,
            tokenizer=tokenizer,
            time_data=None,
            round=args.round,
            exp_type=args.exp,
            qpm_ratio=args.mix_client_qpm_ratio,
            latency_slo=int(slo_value),
            queue_manager=queue_manager  # 传递队列管理器
        )
        clients.append(client)
        tasks.append(client.start())

    # 创建监控器实例
    monitor = ExperimentMonitor(clients, all_results, args.short_clients + args.long_clients + args.mix_clients, args.exp, request_queue,
                                args.use_tunnel)

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor())
    tasks.insert(0, monitor_task)

    # 如果使用队列管理器，启动队列处理（但不加入tasks，让它在后台运行）
    if queue_manager:
        # 队列管理器已经在setup_benchmark_tasks中启动了，这里只需要记录一下
        logger.info(f"Queue manager is running in background with strategy: {queue_manager.strategy.value}")

    return tasks, monitor_task, clients, queue_manager


async def run_benchmark_tasks(tasks, logger):
    """运行基准测试任务"""
    benchmark_timeout = GLOBAL_CONFIG.get('exp_time', 36000)
    
    try:
        await asyncio.wait_for(asyncio.gather(*tasks[1:]), timeout=benchmark_timeout)
    except asyncio.TimeoutError:
        logger.error(f"Tasks did not complete within {benchmark_timeout} seconds, cancelling...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def cancel_monitor_task(monitor_task, logger):
    """取消监控任务"""
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        logger.info("Monitor task cancelled.") 