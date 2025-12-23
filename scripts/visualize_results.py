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
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Install via: pip install matplotlib")

# 策略显示名称和颜色（学术风格配色）
STRATEGY_NAMES = {
    'rr': 'RR',
    'vtc': 'VTC',
    'exfairs': 'ExFairS',
    'justitia': 'Justitia',
    'slo_greedy': 'SLO-Greedy'
}
# 学术风格配色（按图片配色）
STRATEGY_COLORS = {
    'rr': '#e8998d',       # 珊瑚粉色
    'vtc': '#8fb78f',      # 草绿色
    'exfairs': '#8da0cb',  # 蓝紫色
    'justitia': '#e78ac3', # 粉紫色
    'slo_greedy': '#a6d854' # 黄绿色
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
    strategies = ['rr', 'vtc', 'exfairs', 'justitia', 'slo_greedy']
    
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
    
    注意：results.json 中的数据结构：
    - users[user_id]['stats']['avg_total_latency'] (秒)
    - users[user_id]['stats']['p95_latency'] (秒)
    - summary['total_requests']
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
        
        # 计算用户平均延迟（数据在 stats 子字典中，单位是秒）
        avg_latencies = []
        p95_latencies = []
        p99_latencies = []
        
        for user_data in users.values():
            stats = user_data.get('stats', {})
            if stats.get('count', 0) > 0:
                # 延迟单位是秒，转换为毫秒
                avg_lat = stats.get('avg_total_latency', 0) * 1000
                p95_lat = stats.get('p95_latency', 0) * 1000
                p99_lat = stats.get('p99_latency', 0) * 1000
                avg_latencies.append(avg_lat)
                p95_latencies.append(p95_lat)
                p99_latencies.append(p99_lat)
        
        avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0
        
        metrics[strategy] = {
            'completion_rate': (total_completed / total_sent * 100) if total_sent > 0 else 0,
            'slo_violation_rate': (total_slo / total_completed * 100) if total_completed > 0 else 0,
            'timeout_rate': (total_timeout / total_sent * 100) if total_sent > 0 else 0,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': max(p95_latencies) if p95_latencies else 0,
            'p99_latency_ms': max(p99_latencies) if p99_latencies else 0,
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
    strategy_order = ['rr', 'vtc', 'exfairs', 'justitia', 'slo_greedy']
    strategies = [s for s in strategy_order if s in strategies]
    
    colors = [STRATEGY_COLORS.get(s, '#999999') for s in strategies]
    labels = [STRATEGY_NAMES.get(s, s) for s in strategies]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 图1: 基础性能指标 ==========
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle(f'Performance Comparison - {scenario_name}', fontsize=14, fontweight='bold')
    
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
    
    # 3. 平均延迟
    ax = axes1[0, 2]
    values = [metrics[s]['avg_latency_ms'] / 1000 for s in strategies]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Average Latency (s)', fontsize=10)
    ax.set_title('(c) Average Latency ↓', fontsize=10)
    # 自动调整 y 轴范围
    max_val = max(values) if values else 1
    ax.set_ylim(0, max_val * 1.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.05, f'{val:.2f}s', 
                ha='center', va='bottom', fontsize=9)
    
    # 4. Jain Index (SAFI)
    ax = axes1[1, 0]
    values = [metrics[s]['jain_index'] for s in strategies]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Jain Index (SAFI)', fontsize=10)
    ax.set_title('(d) Fairness (SAFI) ↑', fontsize=10)
    min_val = min(values) if values else 0
    max_val = max(values) if values else 1
    y_min = max(0, min_val - 0.1)
    y_max = min(1.01, max_val + 0.1)
    ax.set_ylim(y_min, y_max)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 5. P95/P99 延迟
    ax = axes1[1, 1]
    x = range(len(strategies))
    width = 0.35
    p95_values = [metrics[s]['p95_latency_ms'] / 1000 for s in strategies]
    p99_values = [metrics[s]['p99_latency_ms'] / 1000 for s in strategies]
    
    ax.bar([xi - width/2 for xi in x], p95_values, width, label='P95', 
           color='#7fc97f', edgecolor='black', linewidth=0.5)
    ax.bar([xi + width/2 for xi in x], p99_values, width, label='P99', 
           color='#beaed4', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Latency (s)', fontsize=10)
    ax.set_title('(e) P95/P99 Latency ↓', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    # 自动调整 y 轴
    all_latencies = p95_values + p99_values
    max_lat = max(all_latencies) if all_latencies else 1
    ax.set_ylim(0, max_lat * 1.3)
    
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
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"[✓] Performance chart saved to: {output_path1}")
    
    # ========== 图2: 公平性指标对比（每个策略的 SAFI Jain Index）==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle(f'Fairness Comparison - {scenario_name}', fontsize=14, fontweight='bold')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 每个策略的 Jain Index (SAFI)
    safi_values = [metrics[s]['jain_index'] for s in strategies]
    
    bars = ax2.bar(labels, safi_values, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    
    ax2.set_ylabel('Jain Index (Service Fairness)', fontsize=12)
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.set_title('User SAFI Fairness (Higher is Better)', fontsize=11)
    
    # 设置 y 轴范围：从最小值稍下到 1.0
    min_val = min(safi_values) if safi_values else 0
    ax2.set_ylim(max(0, min_val - 0.1), 1.05)
    
    # 在柱子上显示数值
    for bar, val in zip(bars, safi_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, f"fairness.png")
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"[✓] Fairness chart saved to: {output_path2}")
    
    return output_path1


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
        ('Jain Index (SAFI)', 'jain_index', '', '.4f'),
        ('Jain Index (Token)', 'jain_index_token', '', '.4f'),
        ('Jain Index (SLO Vio)', 'jain_index_slo', '', '.4f'),
        ('Goodput (success)', 'goodput', '', 'd'),
        ('Total Completed', 'total_completed', '', 'd'),
        ('Total Timeout', 'total_timeout', '', 'd'),
        ('Total SLO Violations', 'total_slo_violations', '', 'd'),
    ]
    
    for label, key, suffix, fmt in rows:
        row = f"{label:<30}"
        values = [metrics[s].get(key, 0) for s in strategies]
        
        # 找出最佳值（根据指标类型）
        if key in ['jain_index', 'jain_index_token', 'jain_index_slo', 'completion_rate', 'total_completed', 'goodput']:
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
        
        if chart_path:
            print(f"\n[✓] Visualization complete!")
    else:
        print("\n[!] Install matplotlib for charts: pip install matplotlib")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

