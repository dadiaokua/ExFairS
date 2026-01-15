# ExFairS 文档

## 📚 文档索引

| 文档 | 说明 |
|------|------|
| [快速开始](QUICKSTART.md) | 5分钟快速上手指南 |
| [可视化指南](Visualization_Guide.md) | 结果可视化和分析 |
| [Ubuntu环境设置](UBUNTU_SETUP_GUIDE.md) | Ubuntu系统环境配置 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ExFairS Benchmark                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Clients  │───▶│ RequestQueueMgr  │───▶│ vLLM Engine  │  │
│  └──────────┘    └──────────────────┘    └──────────────┘  │
│       │                  │                                  │
│       │                  ▼                                  │
│       │         ┌──────────────────┐                       │
│       └────────▶│ RealtimeMonitor  │                       │
│                 │  - 统计收集       │                       │
│                 │  - 公平性计算     │                       │
│                 │  - 优先级调整     │                       │
│                 │  - Alpha动态更新  │                       │
│                 └──────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 实时监控流程

```
T0 ─────── T60s ─────── T120s ─────── T180s ─────── ... ─────── T600s
   │         │            │             │                        │
   │ 窗口1   │   窗口2    │   窗口3     │   ...                  │
   │ 收集数据│   收集数据 │   收集数据  │                        │ 结束
   │ 计算SAFI│   计算SAFI │   计算SAFI  │                        │
   │ 更新Alpha│  更新Alpha│   更新Alpha │                        │
   │ 调整优先级│ 调整优先级│  调整优先级 │                        │
   │ 重置窗口│   重置窗口 │   重置窗口  │                        │
```

## 📊 核心概念

### Fairness Ratio

```
fairness_ratio = α × SLO_violation_rate + (1-α) × service_ratio
```

- `α`: 动态调整的权重参数 (0.3-0.95)
- `SLO_violation_rate`: 客户端SLO违约率
- `service_ratio`: 客户端服务量占比

### Jain's Fairness Index

衡量系统公平性的指标，值越接近1表示越公平：

```
J(x) = (Σxi)² / (n × Σxi²)
```

### 优先级交换

当两个客户端的 `fairness_ratio` 差异超过阈值时：
- 体验差的客户端（高 `fairness_ratio`）获得更高优先级
- 体验好的客户端（低 `fairness_ratio`）降低优先级

## 🛠️ 配置文件

### 场景配置 (`config/scenarios/*.yaml`)

```yaml
name: scenario_I
clients:
  - type: Mix
    count: 1
    qpm: 20      # 每分钟请求数
    slo: 20      # SLO目标（秒）
```

### vLLM配置 (`config/vllm/engine_config.yaml`)

```yaml
model_path: "/path/to/model"
gpu_memory_utilization: 0.8
max_num_seqs: 128
tensor_parallel_size: 8
```

### 全局配置 (`config/Config.py`)

```python
GLOBAL_CONFIG = {
    "alpha": 0.8,                    # 初始alpha值
    "fairness_ratio_exfairs": 0.05,  # 交换阈值
    "max_exchange_times": 3,         # 最大交换次数
    "ADJUST_SENSITIVITY": 2.0,       # 调整敏感度
    "priority_amplifier": 10,        # 优先级放大系数
}
```
