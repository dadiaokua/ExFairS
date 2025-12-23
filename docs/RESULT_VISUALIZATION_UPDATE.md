# 结果可视化系统更新总结

## 更新时间
2025-12-23

## 主要改进

### 1. 新的结果存储结构

**旧格式**:
```
results/
  {exp}_{timestamp}.json    # 所有策略混在一起
```

**新格式**:
```
results/
  {run_id}/                 # 批次ID
    {scenario}/             # 场景名称
      {strategy}/           # 策略名称
        results.json        # 详细统计
        config.json         # 实验配置
      charts/               # 可视化图表
        performance.png
        fairness.png
```

**优势**:
- ✅ 清晰的层级结构
- ✅ 便于对比不同策略
- ✅ 便于归档和管理
- ✅ 图表与结果放在一起

### 2. 新的可视化工具

创建了 `scripts/visualize_results.py`，参考了InferSim的可视化模板。

**主要特性**:
- 自动查找最新的实验结果
- 支持指定run_id查看历史结果
- 生成清晰的性能对比图和公平性对比图
- 控制台输出详细的对比表格

**使用方法**:
```bash
# 可视化最新结果
python3 scripts/visualize_results.py scenario_I_balanced

# 指定批次
python3 scripts/visualize_results.py --run-id run_20251223_091625 scenario_I_balanced
```

### 3. 可视化内容

#### 性能对比图 (performance.png)
- 完成率 (Completion Rate)
- SLO违约率 (SLO Violation Rate)
- 平均延迟 (Average Latency)
- Jain公平性指数 (SAFI)
- P95/P99延迟
- Goodput (成功完成的请求数)

#### 公平性对比图 (fairness.png)
- SAFI (Service-Aware Fairness Index)
- Token-based Jain Index
- SLO Violation Jain Index

#### 控制台表格
```
========================================================================================================================
 scenario_I_balanced - Strategy Comparison Summary
========================================================================================================================
Metric                                        RR               VTC           ExFairS          Justitia        SLO-Greedy
------------------------------------------------------------------------------------------------------------------------
Completion Rate                          *95.0%*             95.0%             95.0%             95.0%             95.0%
SLO Violation Rate                         14.9%             14.5%            *8.0%*              8.1%              9.0%
...
```

### 4. 示例数据生成器

创建了 `scripts/generate_sample_results.py` 用于测试可视化功能。

```bash
python3 scripts/generate_sample_results.py
```

生成包含所有策略的示例数据，方便测试和演示。

### 5. 结果处理器更新

更新了 `run_benchmark/result_processor.py`：
- 支持新的结果格式
- 自动保存到结构化目录
- 保留旧格式兼容性

### 6. 参数更新

在 `argument_parser.py` 中新增 `--scenario` 参数：
```python
parser.add_argument('--scenario', type=str, default="default_scenario", 
                   help='Scenario name for result organization')
```

在 `run.sh` 中自动传递scenario参数到Python脚本。

## 文件清单

### 新增文件
- `scripts/visualize_results.py` - 新版可视化工具
- `scripts/generate_sample_results.py` - 示例数据生成器
- `docs/VISUALIZATION_GUIDE.md` - 可视化使用指南

### 修改文件
- `run_benchmark/result_processor.py` - 支持新结果格式
- `run_benchmark/argument_parser.py` - 新增scenario参数
- `run.sh` - 传递scenario参数
- `README.md` - 更新文档，添加可视化说明
- `docs/README.md` - 更新文档索引

### 保留文件（兼容性）
- `plot/plotMain.py` - 旧版可视化工具
- `plot/plotUtil.py` - 旧版工具库

## 使用示例

### 1. 运行实验并生成结果

```bash
# 单场景
./run.sh -e QUEUE_ExFairS --scenario scenario_I_balanced

# 批量运行
./run.sh -s scenario_I,scenario_II -e QUEUE_ExFairS,QUEUE_Justitia
```

结果会保存到 `results/run_{timestamp}/`

### 2. 可视化结果

```bash
# 自动使用最新结果
python3 scripts/visualize_results.py scenario_I_balanced

# 指定特定批次
python3 scripts/visualize_results.py --run-id run_20251223_091625 scenario_I_balanced
```

### 3. 查看图表

图表自动保存到：
```
results/{run_id}/{scenario}/charts/
  - performance.png
  - fairness.png
```

## 向后兼容性

- ✅ 旧的results.json格式仍然被保存
- ✅ 旧的plotMain.py仍然可用
- ✅ 旧的工作流程不受影响
- ✅ 新旧格式可以共存

## 优势总结

1. **更清晰的组织**: 按批次→场景→策略的层级结构
2. **更易对比**: 同一场景下所有策略的结果放在一起
3. **更好的可视化**: 参考InferSim的学术风格图表
4. **更详细的指标**: 三种Jain指数全面评估公平性
5. **更方便的使用**: 自动查找最新结果，一条命令生成所有图表

## 下一步

用户可以：
1. 使用新的可视化工具分析现有结果
2. 参考 `docs/VISUALIZATION_GUIDE.md` 了解详细用法
3. 使用 `scripts/generate_sample_results.py` 生成示例数据测试
4. 继续使用旧的工具（如果习惯的话）

## 技术细节

### results.json 格式

```json
{
  "timestamp": "ISO8601格式",
  "duration": 300.5,
  "strategy": "策略名称",
  "scenario": "场景名称",
  "summary": {
    "total_sent": 总发送数,
    "total_completed": 总完成数,
    "total_slo_violations": SLO违约数,
    "total_timeout": 超时数
  },
  "users": {
    "user_id": {
      "stats": {
        "count": 请求数,
        "avg_total_latency": 平均延迟(秒),
        "p95_latency": P95延迟(秒),
        "p99_latency": P99延迟(秒),
        "avg_queue_latency": 平均排队延迟(秒),
        "successful": 成功数,
        "slo_violations": SLO违约数,
        "timeouts": 超时数
      }
    }
  },
  "fairness": {
    "jain_index_safi": SAFI指数,
    "jain_index_token": Token指数,
    "jain_index_slo_violation": SLO违约指数
  }
}
```

### 可视化工具架构

```
scripts/visualize_results.py
├── find_results_in_run()      # 查找结果文件
├── extract_metrics()          # 提取指标数据
├── plot_comparison()          # 生成对比图
└── print_summary_table()      # 打印表格
```

## 参考

- InferSim可视化模板: `/Users/myrick/GithubProjects/InferSim/scripts/visualize_results.py`
- Matplotlib学术风格配色: ColorBrewer Pastel2 + Set3

