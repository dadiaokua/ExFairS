# InferSim 快速参考卡片

## 🚀 一键命令

```bash
# 使用InferSim模拟器（无需GPU）
./start_vllm_benchmark.sh --use-infersim -e QUEUE_VTC

# 使用真实后端（需要GPU）
./start_vllm_benchmark.sh -e QUEUE_VTC
```

## 📋 常用命令

### InferSim模式

```bash
# 基本使用
./start_vllm_benchmark.sh --use-infersim -e QUEUE_ExFairS

# 指定设备类型
./start_vllm_benchmark.sh --use-infersim --infersim-device H800 -e QUEUE_VTC

# 运行多个实验
./start_vllm_benchmark.sh --use-infersim \
    -e QUEUE_VTC \
    -e QUEUE_FCFS \
    -e QUEUE_ExFairS
```

### 真实后端模式

```bash
# 使用vLLM Engine
./start_vllm_benchmark.sh -e QUEUE_VTC

# 运行多个实验对比
./start_vllm_benchmark.sh \
    -e QUEUE_ExFairS \
    -e QUEUE_VTC \
    -e QUEUE_FCFS
```

## ⚙️ 配置参数

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-infersim` | 启用InferSim | 不启用 |
| `--infersim-device DEVICE` | 设备类型 (H20/H800) | H20 |
| `-e, --exp EXP_NAME` | 实验类型 | QUEUE_ExFairS |
| `-h, --help` | 显示帮助 | - |

### 配置文件参数

编辑 `start_vllm_benchmark.sh`:

```bash
# InferSim配置
INFERSIM_DEVICE="H20"                    # 设备类型: H20 或 H800
INFERSIM_NUM_MACHINES=1                  # 机器数量
INFERSIM_GPUS_PER_MACHINE=8              # 每台机器的GPU数量
INFERSIM_GPUS_PER_MODEL=8                # 模型使用的GPU数量（张量并行大小）
INFERSIM_MAX_BATCH_SIZE=128              # 最大批次大小
INFERSIM_SCHEDULE_INTERVAL=0.05          # 调度间隔（秒）
INFERSIM_USE_FP8_GEMM=0                  # FP8 GEMM优化 (0=否, 1=是)
INFERSIM_USE_FP8_KV=0                    # FP8 KV Cache优化 (0=否, 1=是)
```

**配置说明**:
- **机器数量**: 模拟多机器部署
- **每台机器GPU数**: 每台机器上的GPU卡数
- **模型使用GPU数**: 张量并行大小（Tensor Parallel Size）
- **总GPU数**: `机器数量 × 每台机器GPU数`

**常见配置示例**:

```bash
# 示例1: 单机8卡（默认）
INFERSIM_NUM_MACHINES=1
INFERSIM_GPUS_PER_MACHINE=8
INFERSIM_GPUS_PER_MODEL=8

# 示例2: 单机4卡
INFERSIM_NUM_MACHINES=1
INFERSIM_GPUS_PER_MACHINE=4
INFERSIM_GPUS_PER_MODEL=4

# 示例3: 2机16卡（每机8卡）
INFERSIM_NUM_MACHINES=2
INFERSIM_GPUS_PER_MACHINE=8
INFERSIM_GPUS_PER_MODEL=16

# 示例4: 4机32卡（大模型）
INFERSIM_NUM_MACHINES=4
INFERSIM_GPUS_PER_MACHINE=8
INFERSIM_GPUS_PER_MODEL=32
```

## 🎯 实验类型

### 队列模式（推荐）

| 实验类型 | 说明 |
|---------|------|
| `QUEUE_ExFairS` | ExFairS调度（我们的方法） |
| `QUEUE_VTC` | VTC调度 |
| `QUEUE_FCFS` | FCFS调度 |
| `QUEUE_ROUND_ROBIN` | 轮询调度 |
| `QUEUE_MINQUE` | MinQue调度 |

### 基础模式

| 实验类型 | 说明 |
|---------|------|
| `ExFairS` | ExFairS调度 |
| `VTC` | VTC调度 |
| `FCFS` | FCFS调度 |

## 📊 输出格式

所有后端返回一致的格式：

```python
(
    output_tokens,        # int - 输出token数
    elapsed_time,         # float - 耗时（秒）
    tokens_per_second,    # float - tokens/s
    ttft,                 # float - 首token时间（ms）
    input_tokens,         # int - 输入token数
    slo_met               # int - SLO是否满足 (0/1)
)
```

## 🔍 后端对比

| 后端 | GPU | 启动 | 
|------|-----|------|
| vLLM Engine | ✅ | ~60s |
| HTTP Client | ✅ | ~10s |
| InferSim | ❌ | <1s |

## 📚 文档链接

| 文档 | 链接 |
|------|------|
| 快速开始 | [InferSim_Quick_Start.md](InferSim_Quick_Start.md) |
| 使用指南 | [InferSim_Usage_Guide.md](InferSim_Usage_Guide.md) |
| CLI集成 | [InferSim_CLI_Integration.md](InferSim_CLI_Integration.md) |
| API参考 | [InferSim_API_Reference.md](InferSim_API_Reference.md) |
| 文档索引 | [README.md](README.md) |

## 🐛 常见问题

### 导入失败

```bash
# 确认文件存在
ls -l infersim_benchmark_api.py
```

### 参数错误

```bash
# 查看帮助
./start_vllm_benchmark.sh -h
```

### 性能不符

```bash
# 使用更快的设备
./start_vllm_benchmark.sh --use-infersim --infersim-device H800

# 或启用FP8优化（修改脚本）
INFERSIM_USE_FP8_GEMM=1
```

## 💡 使用技巧

### 快速开发

```bash
# 1. 使用InferSim快速迭代
./start_vllm_benchmark.sh --use-infersim -e NEW_STRATEGY

# 2. 修改代码...

# 3. 再次测试
./start_vllm_benchmark.sh --use-infersim -e NEW_STRATEGY

# 4. 验证（使用真实后端）
./start_vllm_benchmark.sh -e NEW_STRATEGY
```

### 硬件评估

```bash
# 对比不同硬件
for device in H20 H800; do
    ./start_vllm_benchmark.sh --use-infersim \
        --infersim-device $device \
        -e QUEUE_VTC
done
```

### 批量测试

```bash
# 测试所有调度策略
./start_vllm_benchmark.sh --use-infersim \
    -e QUEUE_ExFairS \
    -e QUEUE_VTC \
    -e QUEUE_FCFS \
    -e QUEUE_ROUND_ROBIN \
    -e QUEUE_MINQUE
```

## 🎓 学习路径

1. **5分钟**: 运行第一个InferSim实验
   ```bash
   ./start_vllm_benchmark.sh --use-infersim -e QUEUE_VTC
   ```

2. **10分钟**: 阅读[快速开始](InferSim_Quick_Start.md)

3. **30分钟**: 了解[使用指南](InferSim_Usage_Guide.md)

4. **1小时**: 深入[API参考](InferSim_API_Reference.md)

---

**提示**: 保存这个页面作为快速参考！

