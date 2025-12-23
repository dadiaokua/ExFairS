# ExFairS

**Experiential Fairness Scheduling for Large Language Model Inference Services**

This repository contains the implementation and benchmarking framework for ExFairS, a novel scheduling paradigm that addresses the core gap between system-level metrics and actual user experience in Large Language Model (LLM) inference services. 

## Overview

ExFairS introduces **Experiential Fairness** - a new user-centric approach to fairness optimization in LLM serving systems. Unlike traditional schedulers that focus solely on system-level metrics, ExFairS formulates a composite metric that unifies user SLO (Service Level Objective) compliance with system resource consumption to guide scheduling decisions.

### Key Innovation

- **User-Centric Fairness**: Bridges the gap between system metrics and actual user experience
- **Composite Metric**: Unifies SLO compliance with resource consumption for optimal scheduling
- **Proven Performance**: Reduces SLO violation rates by up to 100% while increasing throughput by 14%-21.9%
- **Concurrent Optimization**: Enhances user experience in concert with, rather than at the expense of, system efficiency

### Research Impact

Our extensive experiments across diverse heterogeneous and high-concurrency workloads demonstrate that ExFairS substantially outperforms state-of-the-art fairness schedulers, providing a quantifiable framework that proves enhancing user experience and improving system efficiency can be achieved simultaneously.

## Features

- **Automatic vLLM Engine Startup**: Built-in vLLM engine management with configurable parameters
- **Multi-Experiment Support**: Run multiple scheduling experiments in sequence with a single command
- **Multiple Scheduling Strategies**: ExFairS, VTC, FCFS, and Queue-based strategies for comprehensive comparison
- **Easy Configuration**: Bash script with command-line argument support
- **Comprehensive Benchmarking**: Test LLM deployments under various concurrency levels and scheduling strategies
- **Advanced Metrics Collection**:
  - Requests per second and latency measurements
  - Tokens per second and time to first token
  - SLO compliance and violation rates
  - Fairness metrics (Jain's Index)
  - User experience quantification
- **Advanced Plotting**: Separate views for performance, fairness, and aggregated metrics
- **JSON Output**: Detailed results for further analysis and visualization

## Requirements

- Python 3.7+
- `openai` Python package
- `numpy` Python package
- `vllm` package (for engine startup)
- `requests` package
- `matplotlib` (for plotting)
- `transformers` (for tokenizer)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dadiaokua/ExFairS.git
   cd ExFairS
   ```

2. Install the required packages:
   ```bash
   pip install openai numpy vllm requests matplotlib transformers
   ```

## Quick Start

### 单场景运行

使用指定场景运行实验：

```bash
# 运行场景 I 的 ExFairS 实验
./run.sh -e QUEUE_ExFairS --scenario scenario_I

# 运行场景 I 的多个策略
./run.sh -e QUEUE_ExFairS,QUEUE_Justitia,QUEUE_SLOGreedy --scenario scenario_I
```

### 批量运行多场景

自动运行多个场景和策略的组合：

```bash
# 运行所有场景的所有策略
./run.sh -s scenario_I,scenario_II,scenario_III,scenario_IV,scenario_V,scenario_VI

# 运行指定场景和指定策略
./run.sh -s scenario_I,scenario_II -e QUEUE_ExFairS,QUEUE_Justitia,QUEUE_SLOGreedy

# 运行单个场景的所有默认策略
./run.sh -s scenario_I
```

### 查看可用选项

```bash
# 列出所有场景
./run.sh --list-scenarios

# 列出所有策略
./run.sh --list-strategies

# 查看完整帮助
./run.sh -h
./start_vllm_benchmark.sh -e ExFairS -e VTC -e FCFS -e QUEUE_ExFairS -e QUEUE_VTC
```

### Getting Help

```bash
./start_vllm_benchmark.sh -h
# or
./start_vllm_benchmark.sh --help
```

## Configuration

The bash script (`start_vllm_benchmark.sh`) contains pre-configured parameters that you can modify:

### Manual Python Execution

For more control, you can run the Python script directly:

```bash
cd run_benchmark
python3 run_benchmarks.py \
    --vllm_url "http://localhost:8000/v1" \
    --api_key "test" \
    --exp "ExFairS" \
    --short_clients 7 \
    --long_clients 3 \
    --round 20 \
    --round_time 300 \
    # ... other parameters
```

### Plotting Results

The system automatically generates three types of plots:

1. **Performance Metrics**: Individual client performance over time
2. **Fairness Metrics**: Fairness ratios, Jain's index, and credits
3. **Aggregated Metrics**: System-wide performance summaries

Plots are saved in the `figure/` directory with timestamps.

## vLLM Engine Management

The benchmark system supports two modes for using vLLM:

### Mode 1: AsyncLLMEngine Direct API (Recommended)

Uses vLLM's AsyncLLMEngine API directly in Python for better performance and resource efficiency:

```bash
python run_benchmarks.py \
  --start_engine True \
  --model_path "/home/llm/model_hub/Qwen2.5-32B-Instruct" \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.9 \
  --vllm_url "http://127.0.0.1:8000"  # Still needed for compatibility, but not used for requests
  # ... other benchmark parameters
```

**Benefits:**
- **Direct Integration**: No HTTP overhead, direct Python API calls
- **Better Performance**: Faster than HTTP requests, no serialization overhead
- **Resource Efficiency**: Shares the same Python process memory space
- **Easier Debugging**: All components run in the same process

### Mode 2: HTTP Server Mode (Traditional)

Uses an external vLLM HTTP server via OpenAI-compatible API:

```bash
# First start vLLM server separately (or use --start_engine False if already running)
python run_benchmarks.py \
  --start_engine False \
  --vllm_url "http://existing-server:8000"
  # ... other benchmark parameters
```

### Automatic Mode Detection

The system automatically detects which mode to use:
- If `--start_engine True`: Uses AsyncLLMEngine direct API
- If `--start_engine False`: Uses HTTP client to connect to external server

### Available vLLM Engine Parameters

- `--start_engine`: Whether to start the vLLM engine (default: True)
- `--model_path`: Path to the vLLM model (default: "/home/llm/model_hub/Qwen2.5-32B-Instruct")
- `--tensor_parallel_size`: Tensor parallel size (default: 8)
- `--pipeline_parallel_size`: Pipeline parallel size (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.9)
- `--max_model_len`: Maximum model length (default: 8124)
- `--max_num_seqs`: Maximum number of sequences (default: 256)
- `--max_num_batched_tokens`: Maximum number of batched tokens (default: 65536)
- `--swap_space`: Swap space size in GB (default: 4)
- `--device`: Device type (default: "cuda")
- `--dtype`: Data type (default: "float16")
- `--quantization`: Quantization method (default: "None")
- `--trust_remote_code`: Trust remote code (default: True)
- `--enable_chunked_prefill`: Enable chunked prefill (default: False)
- `--disable_log_stats`: Disable log statistics (default: False)

### Sampling Parameters (AsyncLLMEngine Mode Only)

When using AsyncLLMEngine direct API mode, you can configure sampling parameters in `config/Config.py`:

```python
GLOBAL_CONFIG = {
    # ... other config ...
    
    # AsyncLLMEngine采样参数
    "sampling_temperature": 0.7,      # Controls randomness (0.0 = deterministic, 1.0+ = more random)
    "sampling_top_p": 0.9,            # Nucleus sampling threshold
    "sampling_top_k": -1,             # Top-k sampling (-1 = disabled)
    "sampling_repetition_penalty": 1.0, # Repetition penalty (1.0 = no penalty)
}
```

## Batch Execution Features

### Sequential Execution
When running multiple experiments:
- Experiments run sequentially, one after another
- Each experiment completes fully before the next begins
- Automatic progress tracking with experiment counters

### Status Reporting
```bash
🚀🚀🚀 开始执行实验 1/3: ExFairS 🚀🚀🚀
[Experiment execution...]
✅ 实验 1/3: ExFairS 已完成

📋 准备开始下一个实验: 2/3 - VTC
==========================================
```

### Final Summary
```bash
🎉🎉🎉 所有实验执行完成！🎉🎉🎉
==========================================
📊 执行结果总览:
  总实验数: 3
  成功实验数: 3
  失败实验数: 0

✅ 成功的实验:
    - ExFairS
    - VTC
    - FCFS
```

## Output and Results

### Result Structure

实验结果采用结构化存储：

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

### Visualization

#### 新版可视化工具（推荐）

```bash
# 可视化最新运行的结果
python3 scripts/visualize_results.py scenario_I_balanced

# 指定特定的运行批次
python3 scripts/visualize_results.py --run-id run_20251223_091625 scenario_I_balanced
```

新版可视化提供：
- 📊 **性能对比图**: 完成率、SLO违约率、延迟、Goodput等
- 📈 **公平性分析**: 三种Jain指数对比（SAFI、Token-based、SLO Violation）
- 📋 **详细表格**: 控制台输出完整的性能对比表

详细说明请参考：[可视化指南](docs/VISUALIZATION_GUIDE.md)

#### 旧版可视化（兼容）

```bash
cd plot
python3 plotMain.py
```

Three types of plots are automatically generated:
1. **Performance plots** (`performance_metrics_*.png`) - System throughput and latency analysis
2. **Fairness plots** (`fairness_metrics_*.png`) - User experience and SLO compliance visualization  
3. **Aggregated plots** (`aggregated_metrics_*.png`) - Comparative scheduler performance

### JSON Results
Results are saved in timestamped JSON files in the `results/` directory:
- Individual client performance metrics and SLO compliance
- ExFairS composite fairness metrics combining user experience and system efficiency
- Comparative analysis against baseline schedulers (VTC, FCFS, etc.)
- System-wide aggregated statistics and throughput measurements


### For Researchers

If you use ExFairS in your research, please cite our paper:

### Coming soon...


## ⚙️ Configuration

### vLLM Engine Configuration

Edit `config/vllm/engine_config.yaml` to modify vLLM parameters:

```yaml
model_path: "/path/to/model"
gpu_memory_utilization: 0.8
max_num_seqs: 128
tensor_parallel_size: 8
```

### Scenario Configuration

Create or edit scenario files in `config/scenarios/`:

```yaml
name: "My_Scenario"
clients:
  short: {count: 2, qpm: [50, 50], slo: [20, 30]}
  long: {count: 2, qpm: [50, 50], slo: [20, 30]}
  mix: {count: 0, qpm: [], slo: []}
experiment:
  round_num: 10
  round_time: 60
  concurrency: 1
```

### Scenario Management Tools

```bash
# List all scenarios
python3 config/scenario_manager.py list

# View scenario details
python3 config/scenario_manager.py show scenario_I

# View vLLM config
python3 config/scenario_manager.py vllm
```

## Scheduling Strategies

This benchmark system supports multiple scheduling strategies for LLM inference:

### Basic Strategies
- **ExFairS** (Experiential Fairness Scheduling): Our proposed scheduler that balances user experience and system efficiency
- **VTC** (Variable Token Credits): Token-based fair scheduling
- **FCFS** (First Come First Serve): Simple FIFO scheduling
- **Justitia**: Virtual time-based fair scheduling with short job prioritization
- **SLOGreedy**: SLO violation rate-based greedy scheduling

### Queue-based Strategies
All basic strategies can be run in queue mode with the `QUEUE_` prefix:
- `QUEUE_ExFairS`, `QUEUE_VTC`, `QUEUE_FCFS`, etc.
- Queue mode provides better resource management and scheduling control
- Supports additional strategies like `QUEUE_ROUND_ROBIN` and `QUEUE_MINQUE`

### Strategy Comparison

| Strategy | Focus | Best For |
|----------|-------|----------|
| **ExFairS** | User experience + efficiency | General-purpose fair scheduling |
| **VTC** | Resource fairness | Balancing token consumption |
| **FCFS** | Simplicity | Baseline comparison |
| **Justitia** | Short job optimization | Mixed workloads with varying lengths |
| **SLOGreedy** | SLO compliance | Meeting service quality objectives |

For detailed explanations of Justitia and SLOGreedy, see [Justitia and SLOGreedy Guide](docs/Justitia_and_SLOGreedy_Guide.md).

## 📂 Project Structure

```
vllm-benchmark/
├── config/
│   ├── vllm/
│   │   └── engine_config.yaml      # vLLM引擎配置
│   ├── scenarios/
│   │   ├── scenario_I.yaml         # 场景I: 均衡负载
│   │   ├── scenario_II.yaml        # 场景II: 不均衡负载
│   │   ├── scenario_III.yaml       # 场景III: 异构4客户端
│   │   ├── scenario_IV.yaml        # 场景IV: 异构8客户端
│   │   ├── scenario_V.yaml         # 场景V: 高并发20客户端
│   │   └── scenario_VI.yaml        # 场景VI: 高并发50客户端
│   └── scenario_manager.py         # 场景管理工具
│
├── results/
│   └── <timestamp>/                # 按时间戳组织的结果目录
│       ├── metadata.json
│       ├── run.log
│       └── <scenario>_<strategy>/  # 具体实验结果
│
├── run_experiments.sh              # 批量运行脚本
├── start_vllm_benchmark.sh         # 单场景运行脚本
└── README.md                       # 本文件
```

## 📖 Additional Documentation

- 📚 [Quick Start Guide](QUICKSTART.md) - 5分钟快速上手
- 📖 [Quick Reference](docs/QUICK_REFERENCE.md) - 常用命令速查
- 📊 [Visualization Guide](docs/Visualization_Guide.md) - 结果可视化指南
- 🔧 [Multi JAIN Index Guide](docs/Multi_JAIN_Index_Enhancement.md) - 公平性指标详解
- 🛠️ [Ubuntu Setup Guide](docs/UBUNTU_SETUP_GUIDE.md) - Ubuntu环境搭建

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

This software is provided for research and educational purposes. Commercial use may require additional licensing considerations.
