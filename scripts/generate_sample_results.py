#!/usr/bin/env python3
"""
生成示例结果用于测试可视化
"""

import json
import os
from pathlib import Path
from datetime import datetime
import random

def generate_sample_results():
    """生成示例结果数据"""
    
    # 创建示例run目录
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sample"
    base_dir = Path(f"results/{run_id}")
    
    # 场景和策略
    scenarios = ["scenario_I_balanced"]
    strategies = ["rr", "vtc", "exfairs", "justitia", "slo_greedy"]
    
    # 用户配置
    users = {
        "user1_short": {"type": "short", "qpm": 50, "slo": 2.0},
        "user2_short": {"type": "short", "qpm": 50, "slo": 2.0},
        "user3_long": {"type": "long", "qpm": 50, "slo": 10.0},
        "user4_long": {"type": "long", "qpm": 50, "slo": 10.0}
    }
    
    for scenario in scenarios:
        for strategy in strategies:
            # 创建目录
            strategy_dir = base_dir / scenario / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成用户统计（不同策略有不同的公平性表现）
            users_stats = {}
            total_sent = 0
            total_completed = 0
            total_slo_violations = 0
            total_timeout = 0
            
            for user_id, user_config in users.items():
                # 根据策略调整统计
                base_requests = 1000
                
                # RR: 所有用户相等
                if strategy == "rr":
                    count = base_requests
                    slo_vio_rate = 0.15
                # VTC: long用户稍好
                elif strategy == "vtc":
                    count = base_requests * (1.1 if "long" in user_id else 0.9)
                    slo_vio_rate = 0.10 if "long" in user_id else 0.20
                # ExFairS: 最公平
                elif strategy == "exfairs":
                    count = base_requests
                    slo_vio_rate = 0.08
                # Justitia: short用户稍好
                elif strategy == "justitia":
                    count = base_requests * (1.1 if "short" in user_id else 0.9)
                    slo_vio_rate = 0.05 if "short" in user_id else 0.12
                # SLO Greedy: 根据SLO动态调整
                else:  # slo_greedy
                    count = base_requests * (1.05 if "long" in user_id else 0.95)
                    slo_vio_rate = 0.09
                
                count = int(count)
                completed = int(count * 0.95)
                slo_violations = int(completed * slo_vio_rate)
                timeouts = count - completed
                
                # 延迟 (秒)
                if "short" in user_id:
                    avg_latency = user_config["slo"] * random.uniform(0.5, 0.8)
                    p95_latency = user_config["slo"] * random.uniform(0.8, 1.1)
                    p99_latency = user_config["slo"] * random.uniform(1.0, 1.3)
                else:
                    avg_latency = user_config["slo"] * random.uniform(0.4, 0.7)
                    p95_latency = user_config["slo"] * random.uniform(0.7, 1.0)
                    p99_latency = user_config["slo"] * random.uniform(0.9, 1.2)
                
                users_stats[user_id] = {
                    "stats": {
                        "count": count,
                        "avg_total_latency": avg_latency,
                        "p95_latency": p95_latency,
                        "p99_latency": p99_latency,
                        "avg_queue_latency": avg_latency * 0.3,
                        "successful": completed,
                        "slo_violations": slo_violations,
                        "timeouts": timeouts
                    }
                }
                
                total_sent += count
                total_completed += completed
                total_slo_violations += slo_violations
                total_timeout += timeouts
            
            # 计算Jain指数（简化计算）
            # ExFairS最公平，RR次之，其他策略较差
            jain_mapping = {
                "exfairs": (0.98, 0.96, 0.94),
                "rr": (0.95, 0.92, 0.88),
                "vtc": (0.88, 0.85, 0.82),
                "justitia": (0.90, 0.87, 0.84),
                "slo_greedy": (0.92, 0.89, 0.86)
            }
            jain_safi, jain_token, jain_slo = jain_mapping[strategy]
            
            # 生成results.json
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "duration": 300.0,
                "strategy": strategy,
                "scenario": scenario,
                "summary": {
                    "total_sent": total_sent,
                    "total_completed": total_completed,
                    "total_slo_violations": total_slo_violations,
                    "total_timeout": total_timeout
                },
                "users": users_stats,
                "fairness": {
                    "jain_index_safi": jain_safi,
                    "jain_index_token": jain_token,
                    "jain_index_slo_violation": jain_slo
                }
            }
            
            # 保存results.json
            with open(strategy_dir / "results.json", 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # 生成config.json
            config_data = {
                "experiment": f"QUEUE_{strategy.upper()}",
                "model": "Qwen2.5-32B-Instruct",
                "dataset": "sharegpt",
                "concurrency": 10,
                "duration": 300,
                "alpha": 0.5,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(strategy_dir / "config.json", 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✓ Generated: {strategy_dir}")
    
    print(f"\n[SUCCESS] Sample results generated in: {base_dir}")
    print(f"\n运行可视化:")
    print(f"  python3 scripts/visualize_results.py --run-id {run_id} scenario_I_balanced")
    
    return run_id

if __name__ == "__main__":
    generate_sample_results()

