# vLLM Benchmark 文档

> 专题文档索引

---

## 📚 核心文档

### 1. [快速开始](QUICKSTART.md) ⭐
> 5分钟快速上手指南

**适合**：新用户快速入门

---

### 2. [快速参考](QUICK_REFERENCE.md) 🔖
> 常用命令和参数速查表

**适合**：日常使用查询

---

### 3. [可视化指南](Visualization_Guide.md) 📊
> 结果图表生成和解读

**适合**：分析实验结果

---

### 4. [修复与改进记录](CRITICAL_FIXES_APPLIED.md) 🔧
> 所有bug修复和性能优化记录

**适合**：了解项目演进和问题解决

---

### 5. [Ubuntu 环境搭建](UBUNTU_SETUP_GUIDE.md) 🖥️
> Ubuntu 系统环境配置

**适合**：Ubuntu 部署

---

## 🚀 快速开始

### 基本使用

```bash
# 单场景运行
./run.sh -e QUEUE_ExFairS --scenario scenario_I

# 批量运行
./run.sh -s scenario_I,scenario_II -e QUEUE_ExFairS,QUEUE_Justitia
```

### 查看帮助

```bash
./run.sh -h                    # 完整帮助
./run.sh --list-scenarios      # 列出场景
./run.sh --list-strategies     # 列出策略
```

---

## 📖 调度策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **QUEUE_ExFairS** | 体验式公平调度 | 综合用户体验和系统效率 |
| **QUEUE_Justitia** | 虚拟时间调度 | 任务长度差异大 |
| **QUEUE_SLOGreedy** | SLO违约率贪心 | 有明确SLO要求 |
| **QUEUE_VTC** | 可变Token积分 | 平衡Token消耗 |
| **QUEUE_FCFS** | 先到先服务 | 基准对比 |

---

## 📊 预定义场景

| 场景 | 描述 | 客户端 | QPM | SLO |
|------|------|--------|-----|-----|
| **I** | 均衡负载 | 2S+2L | 50 (均匀) | 20, 30 |
| **II** | 不均衡负载 | 2S+2L | 10-90 (极化) | 20 |
| **III** | 异构4客户端 | 4 Mix | 20-40 | 15, 20 |
| **IV** | 异构8客户端 | 8 Mix | 10-30 | 10-25 |
| **V** | 高并发20 | 20 Mix | 5-15 | 5-20 |
| **VI** | 高并发50 | 50 Mix | 4 (均匀) | 8-20 |

---

## 🔧 配置管理

### vLLM 配置

编辑 `config/vllm/engine_config.yaml`

### 场景配置

编辑 `config/scenarios/*.yaml` 或创建新场景

### 场景管理工具

```bash
python3 config/scenario_manager.py list           # 列出场景
python3 config/scenario_manager.py show scenario_I  # 查看详情
python3 config/scenario_manager.py vllm           # 查看vLLM配置
```

---

## 📂 结果输出

结果保存在 `results/<timestamp>/` 目录：

```
results/
└── 20251222_120000/
    ├── metadata.json              # 运行元数据
    ├── run.log                    # 运行日志
    └── scenario_I_QUEUE_ExFairS/  # 具体实验结果
        ├── *.json                 # JSON结果
        └── *.png                  # 图表
```

---

## 🔗 相关链接

- [主 README](../README.md) - 项目完整说明
- [修复记录](CRITICAL_FIXES_APPLIED.md) - Bug修复和改进历史

---

## 📝 文档精简说明

本项目文档已精简为 **5个核心文档**：

1. **README.md** (本文件) - 文档索引和快速参考
2. **QUICKSTART.md** - 快速入门指南
3. **QUICK_REFERENCE.md** - 命令速查表
4. **Visualization_Guide.md** - 可视化指南
5. **CRITICAL_FIXES_APPLIED.md** - 修复与改进记录
6. **UBUNTU_SETUP_GUIDE.md** - Ubuntu环境搭建

所有其他文档已整合到以上文档中，避免碎片化。

---

**文档位置**：`docs/` 目录  
**最后更新**：2025-12-23
