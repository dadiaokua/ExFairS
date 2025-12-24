#!/usr/bin/env python3
"""
分析实验阶段时间统计
用法: python3 scripts/analyze_phase_timings.py [run_id]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_run(results_dir: str, run_id: str = None):
    """分析指定运行的阶段时间"""
    results_path = Path(results_dir)
    
    if run_id:
        run_path = results_path / run_id
    else:
        # 获取最新的运行
        runs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
        if not runs:
            print("No runs found")
            return
        run_path = runs[-1]
        run_id = run_path.name
    
    print(f"\n{'='*80}")
    print(f" Phase Timing Analysis - {run_id}")
    print(f"{'='*80}\n")
    
    # 收集所有场景的数据
    all_data = defaultdict(lambda: defaultdict(list))
    
    for scenario_dir in sorted(run_path.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name in ['charts', 'logs']:
            continue
        
        scenario_name = scenario_dir.name
        
        for strategy_dir in sorted(scenario_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            
            strategy_name = strategy_dir.name
            results_file = strategy_dir / 'results.json'
            
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # 获取时间数据
            duration = data.get('duration', 0)
            summary = data.get('summary', {})
            
            all_data[scenario_name][strategy_name] = {
                'duration': duration,
                'total_sent': summary.get('total_sent', 0),
                'total_completed': summary.get('total_completed', 0),
                'total_timeout': summary.get('total_timeout', 0),
                'completion_rate': summary.get('total_completed', 0) / summary.get('total_sent', 1) * 100 if summary.get('total_sent', 0) > 0 else 0
            }
    
    # 打印分析结果
    print(f"{'Scenario':<15} {'Strategy':<15} {'Duration(s)':<12} {'Sent':<8} {'Done':<8} {'Timeout':<8} {'Rate':<8}")
    print("-" * 80)
    
    total_duration = 0
    total_scenarios = 0
    
    for scenario in sorted(all_data.keys()):
        strategies = all_data[scenario]
        for strategy in sorted(strategies.keys()):
            data = strategies[strategy]
            print(f"{scenario:<15} {strategy:<15} {data['duration']:<12.1f} {data['total_sent']:<8} "
                  f"{data['total_completed']:<8} {data['total_timeout']:<8} {data['completion_rate']:<7.1f}%")
            total_duration += data['duration']
        total_scenarios += 1
        print()
    
    print("=" * 80)
    print(f"Total scenarios: {total_scenarios}")
    print(f"Total strategies per scenario: {len(strategies) if strategies else 0}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"Average per strategy: {total_duration / (total_scenarios * len(strategies)) if total_scenarios > 0 and strategies else 0:.1f}s")
    
    return all_data

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment phase timings')
    parser.add_argument('run_id', nargs='?', help='Run ID to analyze (default: latest)')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    args = parser.parse_args()
    
    analyze_run(args.results_dir, args.run_id)

if __name__ == '__main__':
    main()
