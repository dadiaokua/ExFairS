# 结果可视化指南

## 概述

vllm-benchmark 使用结构化的结果存储和强大的可视化工具来帮助您分析实验结果。

## 结果存储结构

### 新格式（推荐）

```
results/
  {run_id}/                         # 批次ID，如 run_20251223_091625
    metadata.json                   # 批次元数据
    {scenario}/                     # 场景名称
      {strategy}/                   # 策略名称
        results.json                # 详细统计数据
        config.json                 # 实验配置
      charts/                       # 可视化图表
        performance.png             # 性能对比图
        fairness.png                # 公平性对比图
```

### results.json 格式

```json
{
  "timestamp": "2025-12-23T09:16:25",
  "duration": 300.5,
  "strategy": "exfairs",
  "scenario": "scenario_I_balanced",
  "summary": {
    "total_sent": 4000,
    "total_completed": 3800,
    "total_slo_violations": 304,
    "total_timeout": 200
  },
  "users": {
    "user1_short": {
      "stats": {
        "count": 1000,
        "avg_total_latency": 1.2,
        "p95_latency": 1.8,
        "p99_latency": 2.1,
        "avg_queue_latency": 0.4,
        "successful": 950,
        "slo_violations": 80,
        "timeouts": 50
      }
    }
  },
  "fairness": {
    "jain_index_safi": 0.98,
    "jain_index_token": 0.96,
    "jain_index_slo_violation": 0.94
  }
}
```

## 使用可视化工具

### 基本用法

```bash
# 可视化最新的运行结果
python3 scripts/visualize_results.py scenario_I_balanced

# 指定特定的运行批次
python3 scripts/visualize_results.py --run-id run_20251223_091625 scenario_I_balanced

# 指定结果目录
python3 scripts/visualize_results.py --results-dir /path/to/results scenario_I_balanced

# 指定输出目录
python3 scripts/visualize_results.py --output-dir /path/to/charts scenario_I_balanced
```

### 命令行参数

- `scenario`: 场景名称（必需），例如 `scenario_I_balanced`
- `--run-id`: 运行批次ID，如果不指定则使用最新的
- `--results-dir`: 结果目录路径（默认: `results`）
- `--output-dir`: 图表输出目录（默认: `results/{run_id}/{scenario}/charts`）

### 输出内容

1. **控制台输出**: 详细的性能对比表格
2. **performance.png**: 性能指标对比图
   - 完成率
   - SLO违约率
   - 平均延迟
   - P95/P99延迟
   - Goodput
   - Jain公平性指数（SAFI）
3. **fairness.png**: 多维度公平性对比
   - SAFI (Service-Aware Fairness Index)
   - Token-based Jain Index
   - SLO Violation Jain Index

## 生成示例数据

用于测试可视化功能：

```bash
python3 scripts/generate_sample_results.py
```

这将生成一个包含所有策略的示例结果集，您可以用它来测试可视化工具。

## 性能指标说明

### 基础指标

- **Completion Rate**: 请求完成率（%）
- **SLO Violation Rate**: SLO违约率（%），越低越好
- **Timeout Rate**: 超时率（%），越低越好
- **Avg Latency**: 平均延迟（毫秒）
- **P95/P99 Latency**: 95/99分位延迟（毫秒）
- **Goodput**: 成功完成的请求数（不含超时）

### 公平性指标

- **Jain Index (SAFI)**: 基于服务感知的公平性指数
  - 计算方式: 基于每个用户的服务比例
  - 范围: 0-1，越接近1越公平
  
- **Jain Index (Token)**: 基于Token数量的公平性指数
  - 计算方式: `input_tokens + 2 * output_tokens`
  - 反映资源消耗的公平性
  
- **Jain Index (SLO Violation)**: 基于SLO违约的公平性指数
  - 计算方式: 基于每个用户的SLO违约率
  - 反映服务质量的公平性

## 与旧版可视化的兼容性

旧版的可视化工具（`plot/plotMain.py`）仍然可用，但新的可视化工具提供了更好的结构和更清晰的展示。

旧版用法:
```bash
cd plot
python3 plotMain.py
```

## 最佳实践

1. **批量运行**: 使用 `run.sh` 的批量模式一次运行多个场景和策略
```bash
   ./run.sh -s scenario_I,scenario_II -e QUEUE_ExFairS,QUEUE_Justitia
   ```

2. **结果对比**: 可视化工具会自动对比同一场景下的所有策略

3. **结果归档**: 每次运行都会创建一个新的 `run_id` 目录，便于管理和对比历史结果

4. **元数据跟踪**: 每个批次都包含 `metadata.json`，记录运行的场景、策略和时间戳

## 故障排查

### matplotlib未安装

如果看到警告 `matplotlib not installed`，请安装：

```bash
pip install matplotlib
```

### 找不到结果

确保：
1. `run_id` 正确
2. 场景名称拼写正确
3. 结果目录包含 `results.json` 文件

### 图表显示异常

检查：
1. 数据完整性（`results.json` 格式正确）
2. matplotlib版本（推荐 >= 3.5.0）
3. 系统字体支持（中文标签可能需要额外配置）
