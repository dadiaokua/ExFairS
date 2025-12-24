#!/usr/bin/env python3
"""
结果处理模块
处理基准测试结果的收集、处理和保存
新格式: results/{run_id}/{scenario}/{strategy}/results.json + config.json
"""

import json
import time
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def save_results_new_format(benchmark_results, fairness_results, args, start_time, end_time, logger):
    """
    保存结果到新格式目录结构
    
    目录结构:
    results/
      {run_id}/
        {scenario}/
          {strategy}/
            results.json  # 包含详细统计数据
            config.json   # 包含实验配置
    
    results.json 格式:
    {
        "timestamp": "2025-12-23T10:30:00",
        "duration": 300.5,
        "summary": {
            "total_sent": 1000,
            "total_completed": 950,
            "total_slo_violations": 50,
            "total_timeout": 50
        },
        "users": {
            "user1": {
                "stats": {
                    "count": 100,
                    "avg_total_latency": 2.5,
                    "p95_latency": 4.2,
                    "p99_latency": 5.8,
                    "avg_queue_latency": 1.0
                }
            }
        },
        "fairness": {
            "jain_index_safi": 0.95,
            "jain_index_token": 0.92,
            "jain_index_slo_violation": 0.88
        }
    }
    """
    # 确定run_id（优先使用传入的run_id，否则使用开始时间）
    if hasattr(args, 'run_id') and args.run_id:
        run_id = args.run_id
        # 确保 run_id 以 'run_' 开头
        if not run_id.startswith('run_'):
            run_id = f"run_{run_id}"
    else:
        run_id = f"run_{datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')}"
    
    # 确定scenario（从args获取）
    scenario = getattr(args, 'scenario', 'default_scenario')
    
    # 确定strategy（从args.exp获取，规范化为可视化脚本期望的名称）
    strategy = args.exp.lower().replace('queue_', '')
    
    # 策略名称映射（统一命名）
    strategy_map = {
        'lfs': 'exfairs',
        'slogreedy': 'slo_greedy',
        'round_robin': 'rr'
    }
    strategy = strategy_map.get(strategy, strategy)
    
    # 创建目录
    result_dir = Path(f"results/{run_id}/{scenario}/{strategy}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算总体统计
    total_sent = 0
    total_completed = 0
    total_slo_violations = 0
    total_timeout = 0
    
    users = {}
    
    # 从benchmark_results中提取数据
    for client_result in benchmark_results:
        if not client_result:
            continue
        
        # 假设每个client_result是一个列表，包含多轮的统计
        # 我们需要汇总每个用户的统计信息
        if not isinstance(client_result, list) or len(client_result) == 0:
            continue
        
        # 获取client信息
        client_info = client_result[0]
        user_id = client_info.get('client_index', 'unknown')
        
        # 汇总该用户的统计
        user_stats = {
            'count': 0,
            'total_latency_sum': 0,
            'latencies': [],
            'queue_latencies': [],
            'slo_violations': 0,
            'timeouts': 0,
            'successful': 0
        }
        
        for round_data in client_result:
            user_stats['count'] += round_data.get('total_requests', 0)
            user_stats['successful'] += round_data.get('successful_requests', 0)
            user_stats['slo_violations'] += round_data.get('slo_violation_count', 0)
            user_stats['timeouts'] += round_data.get('total_requests', 0) - round_data.get('successful_requests', 0)
            
            # 收集延迟数据
            latency_data = round_data.get('latency', {})
            if isinstance(latency_data, dict):
                user_stats['latencies'].append(latency_data.get('p99', 0) / 1000)  # 转换为秒
            
            # 收集排队延迟（新格式支持）
            queue_wait_data = round_data.get('queue_wait_time', {})
            if isinstance(queue_wait_data, dict) and queue_wait_data.get('average', 0) > 0:
                user_stats['queue_latencies'].append(queue_wait_data.get('average', 0))
        
        # 计算平均值
        if user_stats['count'] > 0:
            avg_latency = sum(user_stats['latencies']) / len(user_stats['latencies']) if user_stats['latencies'] else 0
            p95_latency = sorted(user_stats['latencies'])[int(len(user_stats['latencies']) * 0.95)] if user_stats['latencies'] else 0
            p99_latency = sorted(user_stats['latencies'])[int(len(user_stats['latencies']) * 0.99)] if user_stats['latencies'] else 0
            avg_queue_latency = sum(user_stats['queue_latencies']) / len(user_stats['queue_latencies']) if user_stats['queue_latencies'] else 0
            avg_inference_latency = avg_latency - avg_queue_latency if avg_queue_latency > 0 else avg_latency
            
            users[user_id] = {
                'stats': {
                    'count': user_stats['count'],
                    'avg_total_latency': avg_latency,
                    'p95_latency': p95_latency,
                    'p99_latency': p99_latency,
                    'avg_queue_latency': avg_queue_latency,
                    'avg_inference_latency': avg_inference_latency,
                    'successful': user_stats['successful'],
                    'slo_violations': user_stats['slo_violations'],
                    'timeouts': user_stats['timeouts']
                }
            }
            
            # 累加到总体统计
            total_sent += user_stats['count']
            total_completed += user_stats['successful']
            total_slo_violations += user_stats['slo_violations']
            total_timeout += user_stats['timeouts']
    
    # 构建results.json
    results_data = {
        'timestamp': datetime.fromtimestamp(start_time).isoformat(),
        'duration': end_time - start_time,
        'strategy': strategy,
        'scenario': scenario,
        'summary': {
            'total_sent': total_sent,
            'total_completed': total_completed,
            'total_slo_violations': total_slo_violations,
            'total_timeout': total_timeout
        },
        'users': users,
        'fairness': {}
    }
    
    # 添加公平性指标（如果有）
    if fairness_results and len(fairness_results) > 0:
        last_fairness = fairness_results[-1]
        results_data['fairness'] = {
            'jain_index_safi': last_fairness.get('jains_index_safi', last_fairness.get('f_result', 0)),
            'jain_index_token': last_fairness.get('jains_index_token', 0),
            'jain_index_slo_violation': last_fairness.get('jains_index_slo_violation', 0)
        }
    
    # 保存results.json
    results_file = result_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # 构建config.json
    config_data = {
        'experiment': args.exp,
        'model': getattr(args, 'model', 'unknown'),
        'dataset': getattr(args, 'dataset', 'unknown'),
        'concurrency': getattr(args, 'concurrency', 0),
        'duration': getattr(args, 'duration', 0),
        'alpha': getattr(args, 'alpha', 0.5),
        'timestamp': datetime.fromtimestamp(start_time).isoformat()
    }
    
    # 保存config.json
    config_file = result_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved config to {config_file}")
    
    return str(result_dir)


def process_and_save_results(tasks, start_time, args, logger):
    """处理和保存基准测试结果（兼容新旧格式）"""
    all_benchmark_results = []
    for task in tasks[1:]:
        if task.done() and not task.cancelled():
            try:
                result = task.result()
                if result:
                    all_benchmark_results.append(result)
            except Exception as e:
                logger.warning(f"Task result retrieval failed: {e}")

    benchmark_results = all_benchmark_results
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f} seconds")

    # 读取fairness结果
    from config.Config import GLOBAL_CONFIG
    fairness_file_path = f"tmp_result/tmp_fairness_result_{args.exp}_{GLOBAL_CONFIG.get('monitor_file_time')}.json"
    try:
        with open(fairness_file_path, 'r') as f:
            fairness_results = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Fairness results file not found at {fairness_file_path}")
        fairness_results = []

    # 保存到新格式
    result_dir = save_results_new_format(benchmark_results, fairness_results, args, start_time, end_time, logger)
    logger.info(f"Results saved to new format directory: {result_dir}")
    
    # 为了兼容性，也保存旧格式（但保存到新目录结构中）
    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)
    filename = (
        f"{args.exp}_{start_datetime.strftime('%m%d_%H-%M')}_to_{end_datetime.strftime('%H-%M')}.json"
    ).replace(" ", "_").replace(":", "-").replace("/", "-")

    args_dict = vars(args)
    plot_data = {
        "filename": filename,
        "total_time": round(total_time, 2),
        "result_dir": result_dir,  # 传递结构化目录路径
        "figure_dir": result_dir,  # 图表保存目录（与结果同目录）
    }
    plot_data.update(args_dict)
    
    # 保存旧格式到结构化目录（而不是 results/ 根目录）
    from util.FileSaveUtil import save_benchmark_results
    save_benchmark_results(filename, benchmark_results, plot_data, logger, result_dir=result_dir)
    
    return benchmark_results, total_time, filename, plot_data