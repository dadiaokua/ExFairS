#!/usr/bin/env python3
"""
实验结果可视化分析脚本
用法: python3 scripts/visualize_results.py [scenario_name]
      python3 scripts/visualize_results.py scenario_I_balanced
      python3 scripts/visualize_results.py all  # 分析所有场景
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    np = None
    print("[WARNING] matplotlib not installed. Install via: pip install matplotlib")

# 策略显示名称和颜色（学术风格配色）
STRATEGY_NAMES = {
    'rr': 'RR',
    'vtc': 'VTC',
    'exfairs': 'ExFairS',
    'justitia': 'Justitia',
    'slo_greedy': 'SLO-Greedy',
    'fcfs': 'FCFS'
}
# 学术风格配色（按图片配色）
STRATEGY_COLORS = {
    'rr': '#e8998d',       # 珊瑚粉色
    'vtc': '#8fb78f',      # 草绿色
    'exfairs': '#8da0cb',  # 蓝紫色
    'justitia': '#e78ac3', # 粉紫色
    'slo_greedy': '#a6d854', # 黄绿色
    'fcfs': '#fc8d62'      # 橙色
}
# 用户颜色（学术风格浅色）
USER_COLORS = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']


def find_results_in_run(run_dir: str, scenario_name: str) -> dict:
    """
    在批次目录中查找指定场景的结果
    
    目录结构: results/{run_id}/{scenario}/{strategy}/results.json
    例如: results/run_20251217_163946/scenario_I_balanced/rr/results.json
    
    Args:
        run_dir: 批次目录路径（如 results/run_20251217_163946）
        scenario_name: 场景名称（如 scenario_I_balanced）
    
    Returns:
        {strategy: {'data': ..., 'path': ..., 'timestamp': ...}}
    """
    run_path = Path(run_dir)
    scenario_path = run_path / scenario_name
    
    if not scenario_path.exists():
        return {}
    
    results_by_strategy = {}
    strategies = ['rr', 'vtc', 'exfairs', 'justitia', 'slo_greedy', 'fcfs']
    
    for strategy in strategies:
        strategy_path = scenario_path / strategy
        result_file = strategy_path / "results.json"
        
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                results_by_strategy[strategy] = {
                    'data': data,
                    'path': str(strategy_path),
                    'timestamp': data.get('timestamp', '')
                }
            except Exception as e:
                print(f"[WARNING] Failed to load {result_file}: {e}")
    
    return results_by_strategy


def extract_metrics(results: dict) -> dict:
    """
    从结果中提取关键指标
    
    注意：results.json 中的数据结构（所有延迟单位统一为毫秒）：
    - users[user_id]['stats']['avg_total_latency_ms'] (毫秒)
    - users[user_id]['stats']['p95_latency_ms'] (毫秒)
    - users[user_id]['stats']['avg_queue_latency_ms'] (毫秒)
    - users[user_id]['stats']['avg_inference_latency_ms'] (毫秒)
    - summary['total_requests']
    
    兼容旧格式（秒）：
    - users[user_id]['stats']['avg_total_latency'] (秒) -> 转换为毫秒
    """
    metrics = {}
    
    for strategy, result_info in results.items():
        data = result_info['data']
        summary = data.get('summary', {})
        fairness = data.get('fairness', {})
        users = data.get('users', {})
        
        total_sent = summary.get('total_sent', 0)
        total_completed = summary.get('total_completed', 0)
        total_slo = summary.get('total_slo_violations', 0)
        total_timeout = summary.get('total_timeout', 0)
        
        # 计算用户平均延迟（全部毫秒）
        avg_latencies = []
        p95_latencies = []
        p99_latencies = []
        queue_latencies = []
        inference_latencies = []
        
        for user_data in users.values():
            stats = user_data.get('stats', {})
            if stats.get('count', 0) > 0:
                # 优先使用新格式（_ms 后缀，已经是毫秒）
                # 如果没有，使用旧格式（秒，需要 *1000 转换为毫秒）
                if 'avg_total_latency_ms' in stats:
                    # 新格式：直接使用毫秒值
                    avg_lat = stats.get('avg_total_latency_ms', 0)
                    p95_lat = stats.get('p95_latency_ms', 0)
                    p99_lat = stats.get('p99_latency_ms', 0)
                    queue_lat = stats.get('avg_queue_latency_ms', 0)
                    inference_lat = stats.get('avg_inference_latency_ms', 0)
                else:
                    # 旧格式：秒 -> 毫秒
                    avg_lat = stats.get('avg_total_latency', 0) * 1000
                    p95_lat = stats.get('p95_latency', 0) * 1000
                    p99_lat = stats.get('p99_latency', 0) * 1000
                    queue_lat = stats.get('avg_queue_latency', 0) * 1000
                    inference_lat = stats.get('avg_inference_latency', 0) * 1000
                
                avg_latencies.append(avg_lat)
                p95_latencies.append(p95_lat)
                p99_latencies.append(p99_lat)
                queue_latencies.append(queue_lat)
                inference_latencies.append(inference_lat)
        
        avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0
        avg_queue_latency = sum(queue_latencies) / len(queue_latencies) if queue_latencies else 0
        avg_inference_latency = sum(inference_latencies) / len(inference_latencies) if inference_latencies else 0
        
        metrics[strategy] = {
            'completion_rate': (total_completed / total_sent * 100) if total_sent > 0 else 0,
            'slo_violation_rate': (total_slo / total_completed * 100) if total_completed > 0 else 0,
            'timeout_rate': (total_timeout / total_sent * 100) if total_sent > 0 else 0,
            'avg_latency_ms': avg_latency,              # 毫秒
            'avg_queue_latency_ms': avg_queue_latency,  # 毫秒
            'avg_inference_latency_ms': avg_inference_latency,  # 毫秒
            'p95_latency_ms': max(p95_latencies) if p95_latencies else 0,  # 毫秒
            'p99_latency_ms': max(p99_latencies) if p99_latencies else 0,  # 毫秒
            'jain_index': fairness.get('jain_index_safi', fairness.get('jain_index', 0)),
            'jain_index_token': fairness.get('jain_index_token', 0),
            'jain_index_slo': fairness.get('jain_index_slo_violation', 0),
            'goodput': total_completed,  # Goodput = 成功完成的请求数
            'total_completed': total_completed,
            'total_slo_violations': total_slo,
            'total_timeout': total_timeout,
            'users': users
        }
    
    return metrics


def plot_comparison(metrics: dict, scenario_name: str, output_dir: str, results: dict = None):
    """
    生成对比图表（包含性能指标和公平性指标）
    """
    if not HAS_MATPLOTLIB:
        print("[SKIP] Visualization skipped (matplotlib not available)")
        return None
    
    strategies = list(metrics.keys())
    if not strategies:
        print("[WARNING] No data to plot")
        return None
    
    # 确保策略顺序一致
    strategy_order = ['exfairs', 'justitia', 'slo_greedy', 'slogreedy', 'vtc', 'fcfs', 'rr']
    strategies = [s for s in strategy_order if s in strategies]
    
    colors = [STRATEGY_COLORS.get(s, '#999999') for s in strategies]
    labels = [STRATEGY_NAMES.get(s, s) for s in strategies]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 图1: 基础性能指标 ==========
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(f'Performance Comparison - {scenario_name}', fontsize=16, fontweight='bold')
    
    # 设置学术风格
    for ax in axes1.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 1. 完成率
    ax = axes1[0, 0]
    values = [metrics[s]['completion_rate'] for s in strategies]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Completion Rate (%)', fontsize=10)
    ax.set_title('(a) Completion Rate', fontsize=10)
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    # 2. SLO 违约率
    ax = axes1[0, 1]
    values = [metrics[s]['slo_violation_rate'] for s in strategies]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('SLO Violation Rate (%)', fontsize=10)
    ax.set_title('(b) SLO Violation Rate ↓', fontsize=10)
    max_val = max(values) if values else 1
    y_max = max(max_val * 1.3, 1.0)
    ax.set_ylim(0, y_max)
    label_offset = max_val * 0.05 if max_val > 0 else 0.1
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    # 3. 平均延迟（堆叠图：排队时间 + 推理时间，单位转换为秒）
    ax = axes1[0, 2]
    x = np.arange(len(strategies))
    width = 0.6
    
    # 获取总延迟、排队时间（毫秒），转换为秒
    total_times_ms = [metrics[s]['avg_latency_ms'] for s in strategies]
    queue_times_ms = [max(0, metrics[s].get('avg_queue_latency_ms', 0)) for s in strategies]
    inference_times_ms = [max(0, metrics[s].get('avg_inference_latency_ms', 0)) for s in strategies]
    
    # 转换为秒
    total_times_s = [t / 1000 for t in total_times_ms]
    queue_times_s = [q / 1000 for q in queue_times_ms]
    inference_times_s = [i / 1000 for i in inference_times_ms]
    
    # 检查是否有有效的排队时间数据（大于0.001秒=1ms才认为有数据）
    has_queue_data = sum(queue_times_s) > 0.001
    
    if has_queue_data:
        # 绘制堆叠柱状图（排队时间在下，推理时间在上）
        bars1 = ax.bar(x, queue_times_s, width, label='Queue Wait', color='#ff9999', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, inference_times_s, width, bottom=queue_times_s, label='Inference', color='#99ccff', edgecolor='black', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('(c) Latency Breakdown ↓', fontsize=10)
    else:
        # 没有排队数据，显示简单柱状图
        bars = ax.bar(x, total_times_s, width, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('(c) Average Latency ↓', fontsize=10)
    
    ax.set_ylabel('Latency (s)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    
    # 自动调整 y 轴范围（秒）
    max_val_s = max(total_times_s) if total_times_s else 1
    ax.set_ylim(0, max_val_s * 1.3)
    
    # 在柱子上方显示总延迟值（秒）
    for i, total_s in enumerate(total_times_s):
        label = f'{total_s:.2f}s'
        y_pos = total_s + max_val_s * 0.03
        ax.text(i, y_pos, label, ha='center', va='bottom', fontsize=8)
    
    # 4. Jain Index (只显示 SLO 违约率公平性)
    ax = axes1[1, 0]
    x = np.arange(len(strategies))
    
    jain_values = [metrics[s]['jain_index'] for s in strategies]
    
    bars = ax.bar(x, jain_values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Jain Index', fontsize=10)
    ax.set_title('(d) Fairness (SLO Violation) ↑', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    
    # 在柱子上方显示数值
    for i, val in enumerate(jain_values):
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    min_val = min(jain_values) if jain_values else 0
    max_val = max(jain_values) if jain_values else 1
    y_min = max(0, min_val - 0.1)
    y_max = min(1.05, max_val + 0.1)
    ax.set_ylim(y_min, y_max)
    
    # 5. P95/P99 延迟（转换为秒）
    ax = axes1[1, 1]
    x = range(len(strategies))
    width = 0.35
    p95_values_ms = [metrics[s]['p95_latency_ms'] for s in strategies]
    p99_values_ms = [metrics[s]['p99_latency_ms'] for s in strategies]
    
    # 转换为秒
    p95_values_s = [p / 1000 for p in p95_values_ms]
    p99_values_s = [p / 1000 for p in p99_values_ms]
    
    ax.bar([xi - width/2 for xi in x], p95_values_s, width, label='P95', 
           color='#7fc97f', edgecolor='black', linewidth=0.5)
    ax.bar([xi + width/2 for xi in x], p99_values_s, width, label='P99', 
           color='#beaed4', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Latency (s)', fontsize=10)
    ax.set_title('(e) P95/P99 Latency ↓', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    # 自动调整 y 轴（秒）
    all_latencies_s = p95_values_s + p99_values_s
    max_lat_s = max(all_latencies_s) if all_latencies_s else 1
    ax.set_ylim(0, max_lat_s * 1.3)
    
    # 6. Goodput (成功完成的请求数)
    ax = axes1[1, 2]
    values = [metrics[s]['goodput'] for s in strategies]  # 使用正确的 goodput 值
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Goodput (completed)', fontsize=10)
    ax.set_title('(f) Goodput ↑', fontsize=10)
    # 自动调整 y 轴
    max_val = max(values) if values else 1
    ax.set_ylim(0, max_val * 1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, f'{val}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留空间
    output_path1 = os.path.join(output_dir, f"performance.png")
    fig1.savefig(output_path1, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"[✓] Performance chart saved to: {output_path1}")
    
    # 注意：fairness 图已经包含在 performance 图的子图(d)中，不再单独生成
    
    return output_path1


def plot_realtime_trends(scenario_name: str, run_dir: str, output_dir: str):
    """
    生成实时监控模式的趋势图
    
    从 results.json 中读取监控历史数据，绘制：
    1. 各客户端 SLO 违约率变化趋势
    2. Jain Index 变化趋势
    3. Alpha 变化趋势
    4. 各客户端优先级变化趋势
    """
    if not HAS_MATPLOTLIB:
        print("[SKIP] Realtime trends visualization skipped (matplotlib not available)")
        return None
    
    strategies = ['exfairs', 'justitia', 'slo_greedy', 'vtc', 'fcfs', 'rr']
    strategy_data = {}
    
    # 收集每个策略的数据
    for strategy in strategies:
        result_file = Path(run_dir) / scenario_name / strategy / "results.json"
        if not result_file.exists():
            continue
        
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            # 检查是否有监控历史数据
            history = data.get('history', [])
            if not history:
                # 尝试旧格式
                continue
            
            strategy_data[strategy] = {
                'history': history,
                'config': data.get('config', {})
            }
        except Exception as e:
            print(f"[WARNING] Failed to load {result_file}: {e}")
    
    if not strategy_data:
        # 尝试加载旧格式的 benchmark_results.json
        return plot_safi_trends_legacy(scenario_name, run_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个策略生成趋势图
    for strategy, data in strategy_data.items():
        history = data['history']
        
        if not history:
            continue
        
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Realtime Monitor Trends - {scenario_name} - {STRATEGY_NAMES.get(strategy, strategy)}', 
                    fontsize=14, fontweight='bold')
        
        # 提取时间序列
        monitor_points = [h['monitor_count'] for h in history]
        
        # 1. SLO 违约率趋势
        ax = axes[0, 0]
        for i, cid in enumerate(history[0].get('client_stats', {}).keys()):
            slo_rates = [h['client_stats'].get(cid, {}).get('slo_violation_rate', 0) * 100 for h in history]
            color = USER_COLORS[i % len(USER_COLORS)]
            ax.plot(monitor_points, slo_rates, marker='o', markersize=4, 
                   label=cid, color=color, linewidth=1.5)
        ax.set_xlabel('Monitor Point', fontsize=10)
        ax.set_ylabel('SLO Violation Rate (%)', fontsize=10)
        ax.set_title('(a) SLO Violation Rate by Client', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 2. Jain Index 趋势
        ax = axes[0, 1]
        jain_values = [h.get('jain_index', 0) for h in history]
        ax.plot(monitor_points, jain_values, marker='s', markersize=6, 
               color='#8da0cb', linewidth=2, label='Jain Index')
        ax.set_xlabel('Monitor Point', fontsize=10)
        ax.set_ylabel('Jain Index', fontsize=10)
        ax.set_title('(b) Fairness (Jain Index) Trend', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 3. Alpha 变化趋势（仅 ExFairS）
        ax = axes[1, 0]
        if 'alpha' in history[0]:
            alpha_values = [h.get('alpha', 0.8) for h in history]
            ax.plot(monitor_points, alpha_values, marker='^', markersize=6, 
                   color='#e78ac3', linewidth=2, label='Alpha')
            ax.set_ylabel('Alpha', fontsize=10)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'Alpha tracking\nnot available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlabel('Monitor Point', fontsize=10)
        ax.set_title('(c) Alpha Adjustment', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 4. 优先级变化趋势
        ax = axes[1, 1]
        for i, cid in enumerate(history[0].get('client_stats', {}).keys()):
            priorities = [h['client_stats'].get(cid, {}).get('priority', 0) for h in history]
            color = USER_COLORS[i % len(USER_COLORS)]
            ax.plot(monitor_points, priorities, marker='o', markersize=4, 
                   label=cid, color=color, linewidth=1.5)
        ax.set_xlabel('Monitor Point', fontsize=10)
        ax.set_ylabel('Priority', fontsize=10)
        ax.set_title('(d) Priority Changes by Client', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(output_dir, f"realtime_trends_{strategy}.png")
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[✓] Realtime trends chart saved to: {output_path}")
    
    return output_dir


def plot_safi_trends_legacy(scenario_name: str, run_dir: str, output_dir: str):
    """
    旧格式兼容：从 benchmark_results.json 中读取 SAFI 数据
    """
    strategies = ['exfairs', 'justitia', 'slo_greedy', 'vtc', 'fcfs', 'rr']
    strategy_data = {}
    
    for strategy in strategies:
        benchmark_file = Path(run_dir) / scenario_name / strategy / "benchmark_results.json"
        if not benchmark_file.exists():
            continue
        
        try:
            with open(benchmark_file) as f:
                data = json.load(f)
            
            users_data = {}
            for user_idx, user_rounds in enumerate(data):
                if len(user_rounds) > 0:
                    client_id = user_rounds[0].get('client_index', f'user_{user_idx}')
                    fairness_ratios = [round_data.get('fairness_ratio', 0) for round_data in user_rounds]
                    users_data[client_id] = fairness_ratios
            
            if users_data:
                strategy_data[strategy] = users_data
        except Exception as e:
            print(f"[WARNING] Failed to load {benchmark_file}: {e}")
    
    if not strategy_data:
        print("[WARNING] No SAFI data found for any strategy")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_strategies = len(strategy_data)
    n_cols = min(3, n_strategies)
    n_rows = (n_strategies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    fig.suptitle(f'SAFI (Fairness Ratio) Trends - {scenario_name}', fontsize=14, fontweight='bold')
    
    if n_strategies == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, (strategy, users_data) in enumerate(strategy_data.items()):
        ax = axes[idx]
        rounds = list(range(1, len(list(users_data.values())[0]) + 1))
        
        for i, (client_id, fairness_ratios) in enumerate(users_data.items()):
            color = USER_COLORS[i % len(USER_COLORS)]
            ax.plot(rounds, fairness_ratios, marker='o', markersize=4, 
                   label=client_id, color=color, linewidth=1.5)
        
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Fairness Ratio', fontsize=10)
        ax.set_title(f'{STRATEGY_NAMES.get(strategy, strategy)}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlim(0.5, len(rounds) + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for idx in range(len(strategy_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, "safi_trends.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[✓] SAFI trends chart saved to: {output_path}")
    
    return output_path


def print_summary_table(metrics: dict, scenario_name: str):
    """
    打印汇总表格
    """
    print(f"\n{'='*120}")
    print(f" {scenario_name} - Strategy Comparison Summary")
    print(f"{'='*120}")
    
    strategies = ['rr', 'vtc', 'exfairs', 'justitia', 'slo_greedy']
    strategies = [s for s in strategies if s in metrics]
    
    if not strategies:
        print("[WARNING] No results found")
        return
    
    # 表头
    header = f"{'Metric':<30}"
    for s in strategies:
        header += f"{STRATEGY_NAMES.get(s, s):>18}"
    print(header)
    print("-" * 120)
    
    # 数据行
    rows = [
        ('Completion Rate', 'completion_rate', '%', '.1f'),
        ('SLO Violation Rate', 'slo_violation_rate', '%', '.1f'),
        ('Timeout Rate', 'timeout_rate', '%', '.1f'),
        ('Avg Latency (ms)', 'avg_latency_ms', '', '.0f'),
        ('P95 Latency (ms)', 'p95_latency_ms', '', '.0f'),
        ('Jain Index', 'jain_index', '', '.4f'),
        ('Goodput (success)', 'goodput', '', 'd'),
        ('Total Completed', 'total_completed', '', 'd'),
        ('Total Timeout', 'total_timeout', '', 'd'),
        ('Total SLO Violations', 'total_slo_violations', '', 'd'),
    ]
    
    for label, key, suffix, fmt in rows:
        row = f"{label:<30}"
        values = [metrics[s].get(key, 0) for s in strategies]
        
        # 找出最佳值（根据指标类型）
        if key in ['jain_index', 'completion_rate', 'total_completed', 'goodput']:
            best_idx = values.index(max(values)) if values else -1
        else:
            best_idx = values.index(min(values)) if values else -1
        
        for i, (s, v) in enumerate(zip(strategies, values)):
            formatted = f"{v:{fmt}}{suffix}"
            if i == best_idx and best_idx >= 0:
                formatted = f"*{formatted}*"  # 标记最佳
            row += f"{formatted:>18}"
        print(row)
    
    print("-" * 120)
    print("* = Best performance for this metric")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize experiment results"
    )
    parser.add_argument(
        'scenario',
        nargs='?',
        default='latest',
        help='Scenario name (e.g., scenario_I_balanced) or "latest" for most recent results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save charts (default: results/{run_id}/{scenario}/charts)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run batch ID (e.g., run_20251217_163946). If not provided, uses the latest run.'
    )
    
    args = parser.parse_args()
    
    # 查找最新的run_id（如果未指定）
    if not args.run_id:
        results_path = Path(args.results_dir)
        if results_path.exists():
            run_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)
            if run_dirs:
                args.run_id = run_dirs[0].name
                print(f"[INFO] Using latest run: {args.run_id}")
            else:
                print(f"[ERROR] No run directories found in {args.results_dir}")
                return 1
        else:
            print(f"[ERROR] Results directory {args.results_dir} not found")
            return 1
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{args.results_dir}/{args.run_id}/{args.scenario}/charts"
    
    print(f"\n[Visualizing Results]")
    print(f"  Run ID: {args.run_id}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Results Dir: {args.results_dir}")
    print(f"  Output Dir: {output_dir}")
    
    # 查找结果
    run_dir = f"{args.results_dir}/{args.run_id}"
    results = find_results_in_run(run_dir, args.scenario)
    
    if not results:
        print(f"\n[ERROR] No results found for scenario '{args.scenario}'")
        print(f"  Searched in: {run_dir}/{args.scenario}/")
        return 1
    
    print(f"\n  Found {len(results)} strategy results:")
    for strategy, info in results.items():
        print(f"    - {strategy}: {info['path']}")
    
    # 提取指标
    metrics = extract_metrics(results)
    
    # 打印汇总表
    print_summary_table(metrics, args.scenario)
    
    # 生成图表
    if HAS_MATPLOTLIB:
        chart_path = plot_comparison(metrics, args.scenario, output_dir, results)
        
        # 生成实时监控趋势图（兼容旧格式）
        trends_path = plot_realtime_trends(args.scenario, run_dir, output_dir)
        
        if chart_path or trends_path:
            print(f"\n[✓] Visualization complete!")
    else:
        print("\n[!] Install matplotlib for charts: pip install matplotlib")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

