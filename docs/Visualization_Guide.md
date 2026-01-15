# 结果可视化指南

## 概述

ExFairS 使用实时监控模式收集数据，并提供强大的可视化工具来分析实验结果。

## 结果存储结构

```
results/run_<timestamp>/
├── metadata.json                    # 批次元数据
├── run.log                          # 运行日志
└── <scenario>/
    ├── <strategy>/
    │   ├── results.json             # 最终统计结果
    │   ├── benchmark_results.json   # 每次监控的详细数据
    │   ├── config.json              # 实验配置
    │   ├── plot_data.json           # 绘图元数据
    │   ├── realtime_metrics_*.png   # 实时监控趋势图
    │   └── performance_metrics_*.png # 性能指标图
    └── charts/
        ├── performance.png          # 策略对比图
        └── realtime_trends_*.png    # 趋势对比图
```

## 使用可视化工具

### 基本用法

```bash
# 可视化最新的运行结果
python3 scripts/visualize_results.py scenario_I

# 指定运行批次
python3 scripts/visualize_results.py --run-id run_20260116_120000 scenario_I

# 指定输出目录
python3 scripts/visualize_results.py --output-dir ./my_charts scenario_I
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `scenario` | 场景名称 | 必需 |
| `--run-id` | 运行批次ID | 最新批次 |
| `--results-dir` | 结果目录 | `results` |
| `--output-dir` | 输出目录 | 自动生成 |

## 生成的图表

### 1. 实时监控趋势图 (`realtime_metrics_*.png`)

4个子图：
- **(a) SLO Violation Rate Trend**: 各客户端SLO违约率变化
- **(b) Fairness (Jain Index) Trend**: 公平性指数变化
- **(c) Alpha Adjustment**: Alpha参数动态调整（仅ExFairS）
- **(d) Priority Changes**: 各客户端优先级变化

### 2. 性能指标图 (`performance_metrics_*.png`)

4个子图：
- **(a) Latency Breakdown**: 延迟分布（排队时间 + 推理时间）
- **(b) SLO Violation Rate**: 各客户端SLO违约率
- **(c) Request Completion**: 请求完成情况（成功 vs 超时）
- **(d) P95/P99 Latency**: 尾延迟分布

### 3. 策略对比图 (`performance.png`)

6个子图：
- 完成率、SLO违约率、延迟分解
- Jain公平性指数、P95/P99延迟、Goodput

## 性能指标说明

### 基础指标

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| Completion Rate | 请求完成率 | ↑ 越高越好 |
| SLO Violation Rate | SLO违约率 | ↓ 越低越好 |
| Avg Latency | 平均延迟 | ↓ 越低越好 |
| P95/P99 Latency | 尾延迟 | ↓ 越低越好 |
| Goodput | 成功完成请求数 | ↑ 越高越好 |

### 公平性指标

| 指标 | 说明 | 范围 |
|------|------|------|
| Jain Index | Jain公平性指数 | 0-1，越接近1越公平 |
| Alpha | ExFairS动态权重参数 | 0.3-0.95 |

## results.json 格式

```json
{
  "timestamp": "2026-01-16T12:00:00",
  "duration": 600,
  "strategy": "exfairs",
  "scenario": "scenario_I",
  "summary": {
    "total_sent": 500,
    "total_completed": 480,
    "total_slo_violations": 48,
    "total_timeout": 20,
    "monitor_count": 10,
    "final_alpha": 0.75
  },
  "users": {
    "mix_0": {
      "stats": {
        "count": 240,
        "avg_total_latency_ms": 5000,
        "p95_latency_ms": 8000,
        "slo_violations": 24
      }
    }
  },
  "fairness": {
    "jain_index_safi": 0.95,
    "jain_index_slo_violation": 0.92
  },
  "history": [
    {
      "monitor_count": 1,
      "jain_index": 0.85,
      "alpha": 0.8,
      "exchange_count": 2,
      "client_stats": {...}
    }
  ]
}
```

## 故障排查

### matplotlib未安装

```bash
pip install matplotlib
```

### 找不到结果

1. 检查 `run_id` 是否正确
2. 确认场景名称拼写正确
3. 确保结果目录包含 `results.json`

### 图表显示异常

1. 检查数据完整性
2. 确保matplotlib版本 >= 3.5.0
