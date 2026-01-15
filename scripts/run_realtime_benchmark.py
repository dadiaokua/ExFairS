#!/usr/bin/env python3
"""
实时监控模式的基准测试入口

核心特点：
1. 只做一轮，持续时间长（可配置，默认10分钟）
2. 后台监控器每60秒收集一次数据
3. 实时计算SAFI和优先级交换
4. 不阻塞推理，更新后的优先级立即作用于新请求

用法：
    python scripts/run_realtime_benchmark.py --scenario scenario_I --strategy QUEUE_ExFairS --duration 600 --interval 60
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict

# 添加项目根目录到路径 (scripts/ 的父目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # 切换到项目根目录

from config.Config import GLOBAL_CONFIG
from BenchmarkMonitor.RealtimeMonitor import RealtimeMonitor
from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy
from util.BaseUtil import initialize_clients
from util.JsonFormatterUtil import formated_json


def setup_logging(exp_type: str) -> logging.Logger:
    """设置日志"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    GLOBAL_CONFIG["monitor_file_time"] = timestamp
    
    os.makedirs('log', exist_ok=True)
    
    logger = logging.getLogger("RealtimeBenchmark")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        fh = logging.FileHandler(f'log/realtime_benchmark_{exp_type}_{timestamp}.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
    
    return logger


def load_scenario_config(scenario_name: str) -> Dict:
    """加载场景配置"""
    scenario_file = f"config/scenarios/{scenario_name}.yaml"
    
    if not os.path.exists(scenario_file):
        raise FileNotFoundError(f"Scenario config not found: {scenario_file}")
    
    import yaml
    with open(scenario_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_benchmark_clients(scenario_config: Dict, exp_type: str, 
                             queue_manager: RequestQueueManager,
                             result_queue: asyncio.Queue,
                             tokenizer, formatted_json_data: List,
                             openai_clients: List,
                             duration: int) -> List:
    """根据场景配置创建客户端"""
    from BenchmarkClient.BenchmarkClient import BenchmarkClient
    
    clients = []
    client_configs = scenario_config.get('clients', [])
    
    for idx, client_cfg in enumerate(client_configs):
        client_type = client_cfg.get('type', 'Mix')
        count = client_cfg.get('count', 1)
        qpm = client_cfg.get('qpm', 10)
        slo = client_cfg.get('slo', 10.0)
        
        for i in range(count):
            client_index = len(clients)
            
            client = BenchmarkClient(
                client_type=client_type,
                client_index=client_index,
                qpm=qpm,
                port=GLOBAL_CONFIG.get('port', [8000]),
                api_key="empty",
                tokenizer=tokenizer,
                exp_type=exp_type,
                distribution=GLOBAL_CONFIG.get('distribution', 'poisson'),
                request_timeout=GLOBAL_CONFIG.get('request_timeout', 120),
                concurrency=GLOBAL_CONFIG.get('concurrency', 10),
                round=1,
                round_time=duration,
                sleep=0,
                time_data=None,
                result_queue=result_queue,
                formatted_json=formatted_json_data,
                OpenAI_client=openai_clients[client_index % len(openai_clients)],
                qpm_ratio=1.0,
                latency_slo=slo,
                use_time_data=0,
                queue_manager=queue_manager,
                qpm_variation=scenario_config.get('qpm_variation', 0.0),
                qpm_pattern=scenario_config.get('qpm_pattern', 'random'),
                burst_interval=scenario_config.get('burst_interval', 3),
                burst_multiplier=scenario_config.get('burst_multiplier', 2.0)
            )
            
            clients.append(client)
    
    return clients


async def run_realtime_experiment(clients: List, 
                                  queue_manager: RequestQueueManager,
                                  exp_type: str,
                                  monitor_interval: int,
                                  total_duration: int,
                                  logger: logging.Logger) -> tuple:
    """运行实时实验"""
    logger.info("="*60)
    logger.info("Starting Realtime Benchmark Experiment")
    logger.info(f"Experiment Type: {exp_type}")
    logger.info(f"Number of Clients: {len(clients)}")
    logger.info(f"Total Duration: {total_duration}s ({total_duration/60:.1f} min)")
    logger.info(f"Monitor Interval: {monitor_interval}s")
    logger.info("="*60)
    
    # 创建实时监控器
    monitor = RealtimeMonitor(
        clients=clients,
        exp_type=exp_type,
        monitor_interval=monitor_interval,
        total_duration=total_duration
    )
    
    # 为所有客户端设置实时监控器引用
    for client in clients:
        client.realtime_monitor = monitor
    
    # 启动监控器
    await monitor.start()
    
    # 启动所有客户端
    client_tasks = []
    for client in clients:
        task = asyncio.create_task(run_client_continuous(client, total_duration, logger))
        client_tasks.append(task)
    
    logger.info(f"Started {len(client_tasks)} client tasks")
    
    # 等待所有客户端任务完成
    try:
        await asyncio.gather(*client_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error in client tasks: {e}")
    
    # 停止监控器
    await monitor.stop()
    
    # 打印最终结果
    print_final_results(monitor, logger)
    
    return monitor


async def run_client_continuous(client, duration: int, logger: logging.Logger):
    """持续运行客户端"""
    from experiment.queue_experiment import QueueExperiment
    
    start_time = time.time()
    
    strategy_map = {
        "QUEUE_ExFairS": QueueStrategy.PRIORITY,
        "QUEUE_LFS": QueueStrategy.PRIORITY,
        "QUEUE_FCFS": QueueStrategy.FIFO,
        "QUEUE_VTC": QueueStrategy.VTC,
        "QUEUE_Justitia": QueueStrategy.JUSTITIA,
        "QUEUE_SLOGreedy": QueueStrategy.SLO_GREEDY,
        "QUEUE_RR": QueueStrategy.ROUND_ROBIN,
    }
    
    exp_type = client.exp_type
    strategy = strategy_map.get(exp_type, QueueStrategy.PRIORITY)
    
    experiment = QueueExperiment(client, client.queue_manager, strategy)
    await experiment.setup()
    
    config_round = 0
    
    while time.time() - start_time < duration:
        if hasattr(client, 'realtime_monitor') and client.realtime_monitor:
            if not client.realtime_monitor.is_running:
                break
        
        try:
            result = await experiment.run(config_round)
            if result:
                client.results.append(result)
            config_round += 1
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Client {client.client_id}: Error in round {config_round}: {e}")
            await asyncio.sleep(2)
    
    try:
        await experiment.cleanup()
    except Exception as e:
        logger.warning(f"Client {client.client_id}: Cleanup warning: {e}")
    
    logger.info(f"Client {client.client_id}: Finished after {config_round} rounds")


def print_final_results(monitor: RealtimeMonitor, logger: logging.Logger):
    """打印最终结果"""
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    all_stats = monitor.get_all_stats()
    
    if not all_stats:
        logger.warning("No statistics collected")
        return
    
    total_requests = sum(s.cumulative_requests for s in all_stats.values())
    total_successful = sum(s.cumulative_completed for s in all_stats.values())
    total_violations = sum(s.cumulative_slo_violations for s in all_stats.values())
    
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total Requests: {total_requests}")
    logger.info(f"  Successful Requests: {total_successful}")
    logger.info(f"  SLO Violations: {total_violations}")
    logger.info(f"  Overall SLO Violation Rate: {total_violations/total_successful*100:.1f}%" if total_successful > 0 else "  N/A")
    
    logger.info(f"\nPer-Client Statistics:")
    logger.info(f"{'Client ID':<15} {'Requests':<10} {'Completed':<10} {'SLO Vio%':<10} {'Avg Lat(ms)':<12}")
    logger.info("-" * 57)
    
    for cid, stats in all_stats.items():
        slo_rate = stats.cumulative_slo_violation_rate * 100 if stats.cumulative_completed > 0 else 0
        logger.info(f"{cid:<15} {stats.cumulative_requests:<10} {stats.cumulative_completed:<10} "
                   f"{slo_rate:<10.1f} {stats.cumulative_avg_latency_ms:<12.1f}")
    
    if monitor.monitor_history:
        logger.info(f"\nMonitor History ({len(monitor.monitor_history)} points):")
        for record in monitor.monitor_history:
            logger.info(f"  #{record['monitor_count']}: Jain={record['jain_index']:.4f}, "
                       f"Alpha={record['alpha']:.3f}, Exchanges={record['exchange_count']}")


def save_results(monitor: RealtimeMonitor, output_dir: str, 
                args, scenario_config: Dict, logger: logging.Logger):
    """
    保存实验结果（兼容旧格式）
    
    保存的文件：
    - results.json: 最终结果（与旧格式兼容）
    - benchmark_results.json: 每次监控的详细数据（类似旧的每轮数据）
    - config.json: 实验配置
    - plot_data.json: 绘图所需的元数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    start_time_str = GLOBAL_CONFIG.get("monitor_file_time", datetime.now().strftime("%m%d_%H%M"))
    
    # 1. 保存 results.json（与旧格式兼容）
    final_results = monitor.get_final_results()
    final_results["scenario"] = args.scenario
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Saved results.json to {results_file}")
    
    # 2. 保存 benchmark_results.json（每次监控的详细数据）
    # 格式：[[监控点1的各客户端数据], [监控点2的各客户端数据], ...]
    benchmark_results = []
    
    for record in monitor.monitor_history:
        monitor_point_data = []
        for cid, cstats in record.get('client_stats', {}).items():
            # 转换为旧格式
            point_data = {
                "client_id": cid,
                "monitor_count": record['monitor_count'],
                "timestamp": record['timestamp'],
                "elapsed_time": record['elapsed_time'],
                "credit": 0,
                "drift_time": 0,
                "latency_slo": 0,  # 需要从客户端获取
                "slo_violation_count": cstats.get('window_slo_violations', 0),
                "slo_violation_rate": cstats.get('slo_violation_rate', 0),
                "total_requests": cstats.get('window_requests', 0),
                "successful_requests": cstats.get('window_completed', 0),
                "fairness_ratio": cstats.get('fairness_ratio', 0),
                "priority": cstats.get('priority', 0),
                "latency": {
                    "average": cstats.get('avg_latency_ms', 0),
                    "p50": cstats.get('avg_latency_ms', 0),  # 窗口数据没有p50/p95/p99
                    "p95": cstats.get('avg_latency_ms', 0),
                    "p99": cstats.get('avg_latency_ms', 0)
                },
                "time_to_first_token": {
                    "average": cstats.get('avg_ttft_ms', 0),
                    "p50": cstats.get('avg_ttft_ms', 0),
                    "p95": cstats.get('avg_ttft_ms', 0),
                    "p99": cstats.get('avg_ttft_ms', 0)
                },
                "queue_wait_time": {
                    "average": cstats.get('avg_queue_ms', 0),
                    "p50": cstats.get('avg_queue_ms', 0),
                    "p95": cstats.get('avg_queue_ms', 0),
                    "p99": cstats.get('avg_queue_ms', 0)
                },
                "inference_time": {
                    "average": cstats.get('avg_inference_ms', 0),
                    "p50": cstats.get('avg_inference_ms', 0),
                    "p95": cstats.get('avg_inference_ms', 0),
                    "p99": cstats.get('avg_inference_ms', 0)
                },
                "tokens_per_second": {
                    "average": cstats.get('tokens_per_second', 0),
                    "p50": cstats.get('tokens_per_second', 0),
                    "p95": cstats.get('tokens_per_second', 0),
                    "p99": cstats.get('tokens_per_second', 0)
                },
                "jain_index": record.get('jain_index', 0),
                "alpha": record.get('alpha', 0.8),
                "exchange_count": record.get('exchange_count', 0)
            }
            monitor_point_data.append(point_data)
        benchmark_results.append(monitor_point_data)
    
    benchmark_file = os.path.join(output_dir, "benchmark_results.json")
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    logger.info(f"Saved benchmark_results.json to {benchmark_file}")
    
    # 3. 保存 config.json
    config_data = {
        "experiment": args.strategy,
        "model": GLOBAL_CONFIG.get('request_model_name', 'unknown'),
        "dataset": args.dataset,
        "concurrency": GLOBAL_CONFIG.get('concurrency', 10),
        "duration": args.duration,
        "monitor_interval": args.interval,
        "alpha": monitor.alpha,
        "timestamp": timestamp,
        "scenario": args.scenario,
        "clients": scenario_config.get('clients', [])
    }
    
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved config.json to {config_file}")
    
    # 4. 保存 plot_data.json
    end_time_str = datetime.now().strftime("%m%d_%H%M")
    filename = f"{args.strategy}_{start_time_str}_to_{end_time_str}.json"
    
    # 提取客户端配置信息
    mix_qpm = []
    mix_slo = []
    for client_cfg in scenario_config.get('clients', []):
        for _ in range(client_cfg.get('count', 1)):
            mix_qpm.append(client_cfg.get('qpm', 10))
            mix_slo.append(client_cfg.get('slo', 10))
    
    plot_data = {
        "filename": filename,
        "total_time": args.duration,
        "result_dir": output_dir,
        "figure_dir": output_dir,
        "distribution": GLOBAL_CONFIG.get('distribution', 'poisson'),
        "short_qpm": [],
        "short_client_qpm_ratio": 1.0,
        "long_qpm": [],
        "long_client_qpm_ratio": 1.0,
        "mix_qpm": mix_qpm,
        "mix_client_qpm_ratio": 1.0,
        "short_clients": 0,
        "short_clients_slo": [],
        "long_clients": 0,
        "long_clients_slo": [],
        "mix_clients": len(mix_qpm),
        "mix_clients_slo": mix_slo,
        "concurrency": GLOBAL_CONFIG.get('concurrency', 10),
        "request_timeout": GLOBAL_CONFIG.get('request_timeout', 120),
        "exp": args.strategy,
        "scenario": args.scenario,
        "run_id": args.run_id,
        "monitor_interval": args.interval,
        "monitor_count": len(monitor.monitor_history),
        "tokenizer": GLOBAL_CONFIG.get('request_model_name', 'unknown'),
        "request_model_name": GLOBAL_CONFIG.get('request_model_name', 'unknown'),
        "qpm_variation": scenario_config.get('qpm_variation', 0.0),
        "qpm_pattern": scenario_config.get('qpm_pattern', 'random')
    }
    
    plot_data_file = os.path.join(output_dir, "plot_data.json")
    with open(plot_data_file, 'w', encoding='utf-8') as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved plot_data.json to {plot_data_file}")
    
    # 5. 生成可视化图
    generate_visualizations(monitor, output_dir, args.strategy, logger)


def generate_visualizations(monitor: RealtimeMonitor, output_dir: str, 
                           strategy: str, logger: logging.Logger):
    """生成可视化图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        timestamp = GLOBAL_CONFIG.get("monitor_file_time", datetime.now().strftime("%m%d_%H%M"))
        
        # 准备数据
        history = monitor.monitor_history
        if not history:
            logger.warning("No monitoring history to visualize")
            return
        
        monitor_points = [r['monitor_count'] for r in history]
        jain_indices = [r['jain_index'] for r in history]
        alphas = [r['alpha'] for r in history]
        exchange_counts = [r['exchange_count'] for r in history]
        
        # 获取所有客户端ID
        client_ids = list(history[0].get('client_stats', {}).keys()) if history else []
        
        # 每个客户端的 SLO 违约率时间序列
        client_slo_rates = {cid: [] for cid in client_ids}
        client_priorities = {cid: [] for cid in client_ids}
        
        for record in history:
            for cid in client_ids:
                cstats = record.get('client_stats', {}).get(cid, {})
                client_slo_rates[cid].append(cstats.get('slo_violation_rate', 0) * 100)
                client_priorities[cid].append(cstats.get('priority', 0))
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{strategy} - Realtime Monitoring Results', fontsize=14)
        
        # (a) SLO 违约率趋势
        ax1 = axes[0, 0]
        for cid in client_ids:
            ax1.plot(monitor_points, client_slo_rates[cid], marker='o', label=cid, linewidth=2)
        ax1.set_xlabel('Monitor Point')
        ax1.set_ylabel('SLO Violation Rate (%)')
        ax1.set_title('(a) SLO Violation Rate Trend')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # (b) Jain Index 趋势
        ax2 = axes[0, 1]
        ax2.plot(monitor_points, jain_indices, marker='s', color='green', linewidth=2)
        ax2.fill_between(monitor_points, jain_indices, alpha=0.3, color='green')
        ax2.set_xlabel('Monitor Point')
        ax2.set_ylabel('Jain Index')
        ax2.set_title('(b) Fairness (Jain Index) Trend')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # (c) Alpha 和 Exchange Count
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(monitor_points, alphas, marker='o', color='blue', label='Alpha', linewidth=2)
        ax3.set_xlabel('Monitor Point')
        ax3.set_ylabel('Alpha', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3.set_ylim(0, 1)
        
        line2 = ax3_twin.bar(monitor_points, exchange_counts, alpha=0.5, color='orange', label='Exchanges')
        ax3_twin.set_ylabel('Exchange Count', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title('(c) Alpha & Priority Exchanges')
        ax3.grid(True, alpha=0.3)
        
        # (d) 客户端优先级变化
        ax4 = axes[1, 1]
        for cid in client_ids:
            ax4.plot(monitor_points, client_priorities[cid], marker='^', label=cid, linewidth=2)
        ax4.set_xlabel('Monitor Point')
        ax4.set_ylabel('Priority')
        ax4.set_title('(d) Client Priority Changes')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图形
        fig_file = os.path.join(output_dir, f"realtime_metrics_{strategy}_{timestamp}.png")
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved visualization to {fig_file}")
        
        # 额外生成性能指标图
        generate_performance_chart(monitor, output_dir, strategy, timestamp, logger)
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)


def generate_performance_chart(monitor: RealtimeMonitor, output_dir: str,
                               strategy: str, timestamp: str, logger: logging.Logger):
    """生成性能指标图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        all_stats = monitor.get_all_stats()
        if not all_stats:
            return
        
        client_ids = list(all_stats.keys())
        
        # 准备数据
        avg_latencies = [all_stats[cid].cumulative_avg_latency_ms for cid in client_ids]
        avg_queue_times = [all_stats[cid].cumulative_avg_queue_ms for cid in client_ids]
        avg_inference_times = [all_stats[cid].cumulative_avg_inference_ms for cid in client_ids]
        slo_violation_rates = [all_stats[cid].cumulative_slo_violation_rate * 100 for cid in client_ids]
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{strategy} - Performance Metrics', fontsize=14)
        
        x = np.arange(len(client_ids))
        width = 0.35
        
        # (a) 延迟分布（堆叠柱状图）
        ax1 = axes[0, 0]
        ax1.bar(x, avg_queue_times, width, label='Queue Time', color='#3498db')
        ax1.bar(x, avg_inference_times, width, bottom=avg_queue_times, label='Inference Time', color='#e74c3c')
        ax1.set_xlabel('Client')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('(a) Latency Breakdown')
        ax1.set_xticks(x)
        ax1.set_xticklabels(client_ids, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # (b) SLO 违约率
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if rate < 30 else '#f39c12' if rate < 60 else '#e74c3c' for rate in slo_violation_rates]
        ax2.bar(x, slo_violation_rates, width, color=colors)
        ax2.set_xlabel('Client')
        ax2.set_ylabel('SLO Violation Rate (%)')
        ax2.set_title('(b) SLO Violation Rate')
        ax2.set_xticks(x)
        ax2.set_xticklabels(client_ids, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # (c) 请求完成情况
        ax3 = axes[1, 0]
        completed = [all_stats[cid].cumulative_completed for cid in client_ids]
        timeouts = [all_stats[cid].cumulative_requests - all_stats[cid].cumulative_completed for cid in client_ids]
        ax3.bar(x, completed, width, label='Completed', color='#2ecc71')
        ax3.bar(x, timeouts, width, bottom=completed, label='Timeout', color='#e74c3c')
        ax3.set_xlabel('Client')
        ax3.set_ylabel('Request Count')
        ax3.set_title('(c) Request Completion')
        ax3.set_xticks(x)
        ax3.set_xticklabels(client_ids, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # (d) P95/P99 延迟
        ax4 = axes[1, 1]
        p95 = [all_stats[cid].p95_latency_ms for cid in client_ids]
        p99 = [all_stats[cid].p99_latency_ms for cid in client_ids]
        ax4.bar(x - width/2, p95, width, label='P95', color='#3498db')
        ax4.bar(x + width/2, p99, width, label='P99', color='#9b59b6')
        ax4.set_xlabel('Client')
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('(d) P95/P99 Latency')
        ax4.set_xticks(x)
        ax4.set_xticklabels(client_ids, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        fig_file = os.path.join(output_dir, f"performance_metrics_{strategy}_{timestamp}.png")
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved performance chart to {fig_file}")
        
    except Exception as e:
        logger.error(f"Error generating performance chart: {e}", exc_info=True)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run realtime benchmark experiment')
    
    parser.add_argument('--scenario', type=str, default='scenario_I',
                        help='Scenario name (e.g., scenario_I, scenario_II)')
    parser.add_argument('--strategy', type=str, default='QUEUE_ExFairS',
                        help='Scheduling strategy')
    parser.add_argument('--duration', type=int, default=600,
                        help='Total experiment duration in seconds (default: 600)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Monitor interval in seconds (default: 60)')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port (default: 8000)')
    parser.add_argument('--model', type=str, default='',
                        help='Model name/path for tokenizer')
    parser.add_argument('--dataset', type=str, default='sharegpt',
                        help='Dataset to use (default: sharegpt)')
    parser.add_argument('--run-id', type=str, default='',
                        help='Run ID for organizing results')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory for results')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(args.strategy)
    
    logger.info("="*60)
    logger.info("Realtime Benchmark Configuration")
    logger.info(f"  Scenario: {args.scenario}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Duration: {args.duration}s ({args.duration/60:.1f} min)")
    logger.info(f"  Monitor Interval: {args.interval}s")
    logger.info(f"  Port: {args.port}")
    logger.info("="*60)
    
    # 更新全局配置
    GLOBAL_CONFIG['port'] = [args.port]
    if args.model:
        GLOBAL_CONFIG['request_model_name'] = args.model
    
    # 生成 run_id
    if not args.run_id:
        args.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 加载场景配置
    try:
        scenario_config = load_scenario_config(args.scenario)
        logger.info(f"Loaded scenario config: {args.scenario}")
    except Exception as e:
        logger.error(f"Failed to load scenario config: {e}")
        return
    
    # 初始化分词器
    try:
        from transformers import AutoTokenizer
        model_name = GLOBAL_CONFIG.get('request_model_name', 'meta-llama/Llama-2-7b-hf')
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Loaded tokenizer for model: {model_name}")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}, using fallback")
        tokenizer = lambda x, **kwargs: type('obj', (object,), {'input_ids': [[0]*len(x.split())]})()
    
    # 加载数据集
    try:
        formatted_json_data = formated_json(args.dataset, "default", tokenizer)
        logger.info(f"Loaded dataset: {args.dataset} ({len(formatted_json_data)} samples)")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 初始化OpenAI客户端
    try:
        openai_clients = initialize_clients(args.port)
        logger.info(f"Initialized {len(openai_clients)} OpenAI clients")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI clients: {e}")
        return
    
    # 创建结果队列
    result_queue = asyncio.Queue()
    
    # 创建队列管理器
    queue_manager = RequestQueueManager()
    queue_manager.set_openai_client(openai_clients)
    
    # 启动队列管理器
    queue_processing_task = asyncio.create_task(queue_manager.start_processing(num_workers=10))
    await asyncio.sleep(2)
    
    if not queue_manager.is_running:
        logger.error("Failed to start queue manager")
        return
    
    logger.info("Queue manager started successfully")
    
    # 创建客户端
    clients = create_benchmark_clients(
        scenario_config=scenario_config,
        exp_type=args.strategy,
        queue_manager=queue_manager,
        result_queue=result_queue,
        tokenizer=tokenizer,
        formatted_json_data=formatted_json_data,
        openai_clients=openai_clients,
        duration=args.duration
    )
    
    logger.info(f"Created {len(clients)} benchmark clients")
    
    # 运行实时实验
    try:
        monitor = await run_realtime_experiment(
            clients=clients,
            queue_manager=queue_manager,
            exp_type=args.strategy,
            monitor_interval=args.interval,
            total_duration=args.duration,
            logger=logger
        )
        
        # 确定结果保存路径
        if args.output_dir:
            output_dir = args.output_dir
        else:
            strategy_name = args.strategy.replace("QUEUE_", "").lower()
            output_dir = f"results/{args.run_id}/{args.scenario}/{strategy_name}"
        
        # 保存结果
        save_results(monitor, output_dir, args, scenario_config, logger)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        # 停止队列管理器
        await queue_manager.stop()
        queue_processing_task.cancel()
        try:
            await queue_processing_task
        except asyncio.CancelledError:
            pass
        
        logger.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
