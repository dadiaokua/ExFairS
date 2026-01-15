# ExFairS 快速开始

> 5分钟快速上手指南

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/dadiaokua/ExFairS.git
cd ExFairS

# 安装依赖
pip install -r requirements.txt
```

## 🚀 运行实验

### 快速运行（实时监控模式）

```bash
# 运行单个场景单个策略
./run.sh --scenario scenario_I --strategy QUEUE_ExFairS

# 运行单个场景多个策略对比
./run.sh --scenario scenario_I
```

### 批量运行

```bash
# 运行所有场景（推荐）
./run.sh --all

# 运行指定场景
./run.sh --scenario scenario_I,scenario_II,scenario_III
```

### 自定义参数

```bash
# 自定义持续时间和监控间隔
./run.sh --scenario scenario_I --duration 600 --interval 60
```

## 🎯 调度策略

| 策略 | 说明 |
|------|------|
| `QUEUE_ExFairS` | **体验式公平调度**（我们的方法） |
| `QUEUE_Justitia` | 虚拟时间调度（短任务优先） |
| `QUEUE_SLOGreedy` | SLO违约率贪心调度 |
| `QUEUE_VTC` | 可变Token积分 |
| `QUEUE_FCFS` | 先到先服务 |

## 📊 预定义场景

| 场景 | 说明 | 客户端配置 |
|------|------|-----------|
| `scenario_I` | 均衡负载 | 2 Mix客户端 |
| `scenario_II` | 不均衡负载 | 2 Mix客户端 |
| `scenario_III` | 异构4客户端 | 4 Mix客户端 |
| `scenario_IV` | 异构8客户端 | 8 Mix客户端 |
| `scenario_V` | 高并发 | 20 Mix客户端 |

## 📂 查看结果

实验结果保存在 `results/run_<timestamp>/` 目录：

```
results/run_20260116_120000/
├── metadata.json                    # 运行元数据
├── run.log                          # 运行日志
└── scenario_I/
    ├── exfairs/
    │   ├── results.json             # 最终统计
    │   ├── benchmark_results.json   # 监控历史
    │   └── *.png                    # 可视化图
    ├── fcfs/
    ├── vtc/
    └── charts/                      # 对比图表
```

## 🔍 常用命令

```bash
# 查看帮助
./run.sh --help

# 列出所有场景
./run.sh --list-scenarios

# 列出所有策略
./run.sh --list-strategies

# 可视化结果
python3 scripts/visualize_results.py scenario_I
```

## ⚙️ 配置修改

### vLLM 引擎配置

编辑 `config/vllm/engine_config.yaml`：

```yaml
model_path: "/path/to/your/model"
gpu_memory_utilization: 0.8
max_num_seqs: 128
tensor_parallel_size: 8
```

### 场景配置

编辑 `config/scenarios/scenario_I.yaml`：

```yaml
name: scenario_I
clients:
  - type: Mix
    count: 1
    qpm: 20
    slo: 20
  - type: Mix
    count: 1
    qpm: 60
    slo: 20
```

## 📚 更多文档

- [可视化指南](Visualization_Guide.md)
- [Ubuntu环境设置](UBUNTU_SETUP_GUIDE.md)
