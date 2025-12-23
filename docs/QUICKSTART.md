# vLLM Benchmark 快速开始

> 5分钟快速上手指南

---

## 📦 安装

```bash
# 克隆项目
git clone <repository-url>
cd vllm-benchmark

# 安装依赖
pip install -r requirements.txt

# 或使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 运行第一个实验

### 方法 1：使用默认配置

```bash
./run.sh -e QUEUE_ExFairS
```

### 方法 2：指定场景

```bash
./run.sh -e QUEUE_ExFairS --scenario scenario_I
```

### 方法 3：批量运行（推荐）

```bash
# 运行所有场景和默认策略
./run.sh -s scenario_I,scenario_II,scenario_III,scenario_IV,scenario_V,scenario_VI

# 运行特定场景和策略
./run.sh -s scenario_I,scenario_II -e QUEUE_ExFairS,QUEUE_Justitia,QUEUE_SLOGreedy
```

---

## 🎯 可用的调度策略

| 策略名称 | 说明 |
|---------|------|
| `QUEUE_ExFairS` | 体验式公平调度 |
| `QUEUE_Justitia` | 虚拟时间调度（短任务优先） |
| `QUEUE_SLOGreedy` | SLO违约率贪心调度 |
| `QUEUE_VTC` | 可变Token积分 |
| `QUEUE_FCFS` | 先到先服务 |
| `QUEUE_MINQUE` | QuE调度 |

---

## 📊 预定义场景

| 场景 | 说明 | 客户端 |
|------|------|--------|
| `scenario_I` | 均衡负载 | 2S+2L, QPM=50 |
| `scenario_II` | 不均衡负载 | 2S+2L, QPM=10-90 |
| `scenario_III` | 异构4客户端 | 4 Mix, QPM=20-40 |
| `scenario_IV` | 异构8客户端 | 8 Mix, QPM=10-30 |
| `scenario_V` | 高并发20 | 20 Mix, QPM=5-15 |
| `scenario_VI` | 高并发50 | 50 Mix, QPM=4 |

---

## 📂 查看结果

结果保存在 `results/<timestamp>/` 目录：

```
results/
└── 20251222_120000/
    ├── metadata.json              # 运行元数据
    ├── run.log                    # 运行日志
    └── scenario_I_QUEUE_ExFairS/  # 实验结果
        ├── *.json                 # JSON结果
        └── *.png                  # 图表
```

---

## ⚙️ 配置修改

### 修改 vLLM 参数

编辑 `config/vllm/engine_config.yaml`：

```yaml
model_path: "/path/to/your/model"
gpu_memory_utilization: 0.8
max_num_seqs: 128
```

### 创建自定义场景

复制并修改场景文件：

```bash
cp config/scenarios/scenario_I.yaml config/scenarios/my_scenario.yaml
# 编辑 my_scenario.yaml
./run.sh -e QUEUE_ExFairS --scenario my_scenario
```

---

## 🔍 常用命令

```bash
# 列出所有场景
./run.sh --list-scenarios

# 列出所有策略
./run.sh --list-strategies

# 查看场景详情
python3 config/scenario_manager.py show scenario_I

# 查看帮助
./run.sh -h
```

---

## 📚 下一步

- [完整文档](docs/README.md)
- [完整指南](docs/Complete_Guide.md)
- [项目结构](docs/Project_Structure.md)
- [变更说明](CHANGES.md)

---

**提示**：首次运行可能需要下载模型和数据集，请耐心等待。

