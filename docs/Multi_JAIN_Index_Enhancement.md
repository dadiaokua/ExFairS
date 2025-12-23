# 多维度 JAIN 公平性指数增强

## 📊 概述

在原有基础上，实验现在计算**三种不同维度**的 JAIN 公平性指数（Jain's Fairness Index），更全面地评估系统的公平性表现。

## 🎯 三种 JAIN 指数

### 1️⃣ SAFI (Service-Aware Fairness Index)
**基于 `fairness_ratio` 计算**

```python
fairness_ratio = service_ratio * (1 - alpha) + alpha * slo_violation_ratio
```

- **含义**：综合服务量和 SLO 违约率的加权公平性指标
- **服务量**：`input_tokens + 2 * output_tokens`
- **权重参数**：`alpha` 控制 SLO 违约的重要性（默认 0.5）
- **评估维度**：整体服务质量的公平分配

### 2️⃣ Token-based JAIN Index
**基于实际 token 数量计算**

```python
token_value = total_input_tokens + 2 * total_output_tokens
```

- **含义**：衡量各客户端获得的 token 资源是否公平分配
- **评估维度**：计算资源（token 生成）的公平分配
- **输出权重 2x**：因为输出 token 生成成本更高

### 3️⃣ SLO Violation JAIN Index
**基于 SLO 违约比例计算**

```python
slo_violation_ratio = slo_violation_count / total_requests
```

- **含义**：衡量各客户端的延迟体验是否公平
- **评估维度**：服务质量承诺（SLO）的公平性
- **越接近 1**：各客户端的 SLO 违约率越均衡

## 🔧 技术实现

### 核心改进

#### 1. 泛化的 JAIN Index 计算函数

```python
def calculate_Jains_index(clients, exp_type, metric_name="fairness_ratio", values=None):
    """
    Args:
        clients: 客户端列表
        exp_type: 实验类型
        metric_name: 指标名称（用于日志记录）
        values: 可选的值列表。如果为 None，使用 client.fairness_ratio
        
    支持两种指标方向：
    - "smaller is better": fairness_ratio, slo_violation_ratio
    - "larger is better": token_count
    """
```

**关键特性**：
- ✅ 自动识别指标方向（越大越好 vs 越小越好）
- ✅ 动态归一化和转换
- ✅ 详细的计算日志记录

#### 2. 修改 `fairness_result` 函数

```python
async def fairness_result(clients, exp_type, logger):
    # ... 计算服务值和公平性比率 ...
    
    # 计算三种 JAIN 指数
    safi_jains_index = calculate_Jains_index(clients, exp_type, metric_name="SAFI_fairness_ratio")
    
    token_values = [result["total_input_tokens"] + 2 * result["total_output_tokens"] 
                    for client in clients for result in [client.results[-1]]]
    token_jains_index = calculate_Jains_index(clients, exp_type, metric_name="token_count", values=token_values)
    
    slo_violation_ratios = [client.slo_violation_count / client.results[-1]['total_requests'] 
                            for client in clients]
    slo_jains_index = calculate_Jains_index(clients, exp_type, metric_name="slo_violation_ratio", values=slo_violation_ratios)
    
    # 返回字典格式
    return {
        "safi": safi_jains_index,
        "token": token_jains_index,
        "slo_violation": slo_jains_index
    }, service
```

#### 3. 更新结果保存格式

**新的 JSON 格式**：
```json
{
  "jains_index_safi": 0.8542,
  "jains_index_token": 0.9123,
  "jains_index_slo_violation": 0.7834,
  "f_result": 0.8542,  // 保持向后兼容
  "s_result": [...],
  "time": "2024-12-09 15:30:00",
  "exchange_count": 3
}
```

**向后兼容**：
- 保留 `f_result` 字段（值为 SAFI）
- 旧代码仍可正常工作
- 新代码可访问所有三个指标

#### 4. 可视化增强

**绘图改进**（`plot/plotMain.py`）：

```python
# 同时绘制三条 JAIN index 曲线
axs2[1].plot(times, safi_values, marker='o', label='SAFI (Service-Aware)', linewidth=2)
axs2[1].plot(times, token_values, marker='s', label='Token-based', linewidth=2)
axs2[1].plot(times, slo_values, marker='^', label='SLO Violation', linewidth=2)
```

**新的图表特性**：
- 📈 三条独立曲线，不同标记符号
- 🎨 清晰的图例标识
- 📊 网格线和标签优化

## 📈 使用示例

### 运行实验

```bash
# 正常运行实验，新的 JAIN 指数会自动计算
./start_vllm_benchmark.sh --exp-type QUEUE_ExFairS --use-infersim
```

### 查看日志

```bash
# 日志会包含三个 JAIN 指数
2024-12-09 15:30:00 - ExperimentMonitor-QUEUE_ExFairS - INFO - Fairness calculation complete:
2024-12-09 15:30:00 - ExperimentMonitor-QUEUE_ExFairS - INFO -   SAFI JAIN: 0.8542
2024-12-09 15:30:00 - ExperimentMonitor-QUEUE_ExFairS - INFO -   Token JAIN: 0.9123
2024-12-09 15:30:00 - ExperimentMonitor-QUEUE_ExFairS - INFO -   SLO Violation JAIN: 0.7834
```

### 查看结果文件

```bash
# 查看详细的 JAIN 计算日志
cat tmp_result/QUEUE_ExFairS_jains_index_calculation_log_12_09_15_30.log

# 查看结果 JSON
cat tmp_result/tmp_fairness_result_QUEUE_ExFairS_12_09_15_30.json
```

### 生成可视化

```bash
# 绘图时会自动显示三条 JAIN 曲线
python plot/plotMain.py
```

## 📊 解读指标

### JAIN 指数范围
- **1.0**: 完美公平（所有客户端值完全相同）
- **0.0**: 完全不公平（极端不均衡）
- **越接近 1**: 公平性越好

### 三个维度的含义

| 指标 | 高值表示 | 低值表示 | 适用场景 |
|------|----------|----------|----------|
| **SAFI** | 综合服务质量公平 | 某些客户端获得更多服务或更好体验 | 整体公平性评估 |
| **Token** | token 资源分配均衡 | 某些客户端消耗更多计算资源 | 资源使用公平性 |
| **SLO Violation** | SLO 违约率均衡 | 某些客户端延迟体验明显更差 | 用户体验公平性 |

### 典型场景分析

**场景 1: 高 SAFI，低 Token**
```
SAFI: 0.92, Token: 0.65, SLO: 0.88
```
→ 虽然综合服务质量公平，但资源消耗不均衡（可能短请求客户端数量多但总 token 少）

**场景 2: 高 Token，低 SLO**
```
SAFI: 0.78, Token: 0.90, SLO: 0.60
```
→ token 资源分配均衡，但部分客户端延迟体验差（可能长请求被延误）

**场景 3: 三者均衡**
```
SAFI: 0.88, Token: 0.85, SLO: 0.90
```
→ 理想状态，各维度公平性都很好

## 🔍 详细计算过程

### JAIN 指数计算公式

给定 n 个客户端的值 x₁, x₂, ..., xₙ：

```
归一化: normalized_i = (x_i - min) / (max - min)

转换（如果越小越好）: 
    transformed_i = 1 - normalized_i
    
转换（如果越大越好）: 
    transformed_i = normalized_i

JAIN Index = (Σ transformed_i)² / (n × Σ transformed_i²)
```

### 示例计算

假设有 3 个客户端的 token 数量：
- Client A: 1000 tokens
- Client B: 1500 tokens  
- Client C: 2000 tokens

**步骤 1: 归一化**
```
min = 1000, max = 2000
normalized_A = (1000 - 1000) / (2000 - 1000) = 0.0
normalized_B = (1500 - 1000) / (2000 - 1000) = 0.5
normalized_C = (2000 - 1000) / (2000 - 1000) = 1.0
```

**步骤 2: 转换（token 是越大越好）**
```
transformed_A = 0.0
transformed_B = 0.5
transformed_C = 1.0
```

**步骤 3: 计算 JAIN**
```
sum = 0.0 + 0.5 + 1.0 = 1.5
sum_squares = 0.0² + 0.5² + 1.0² = 1.25
JAIN = (1.5)² / (3 × 1.25) = 2.25 / 3.75 = 0.6
```

→ 0.6 表示中等公平性（存在一定不均衡）

## 📝 代码修改清单

### 修改的文件

1. **`util/MathUtil.py`**
   - ✅ 泛化 `calculate_Jains_index` 函数
   - ✅ 修改 `fairness_result` 返回多个 JAIN 指数

2. **`BenchmarkMonitor/BenchmarkMonitor.py`**
   - ✅ 更新 `_process_complete_round` 处理字典格式
   - ✅ 更新 `_save_results` 保存多个指标
   - ✅ 增强日志输出

3. **`util/FileSaveUtil.py`**
   - ✅ 更新 `save_results` 支持新旧格式
   - ✅ 向后兼容性保证

4. **`plot/plotMain.py`**
   - ✅ 更新 `plot_fairness_results` 绘制三条曲线
   - ✅ 向后兼容旧格式数据

### 新增内容

- ✅ 多维度公平性评估
- ✅ Token 资源公平性指标
- ✅ SLO 违约公平性指标
- ✅ 详细的计算日志
- ✅ 增强的可视化

### 保持不变

- ✅ 公平性调整逻辑（仍基于 SAFI）
- ✅ 资源交换机制
- ✅ 优先级调整策略
- ✅ 所有实验类型支持

## 🚀 优势

### 1. 更全面的公平性评估
不再依赖单一指标，从多个角度评估系统公平性。

### 2. 更深入的性能洞察
可以识别不同维度的不公平现象：
- 资源分配不均（Token JAIN 低）
- 体验质量不均（SLO JAIN 低）
- 综合服务不均（SAFI 低）

### 3. 更好的调优指导
三个指标可以指导不同的优化方向：
- Token JAIN 低 → 优化资源分配策略
- SLO JAIN 低 → 优化调度优先级
- SAFI 低 → 调整 alpha 权重

### 4. 向后兼容
旧的实验结果和代码无需修改仍可工作。

## 🎯 未来扩展

可能的扩展方向：
- ✨ 增加 TTFT（首Token时间）公平性指标
- ✨ 增加吞吐量公平性指标
- ✨ 支持自定义加权的组合指标
- ✨ 实时公平性监控和告警

## 📚 参考

- [Jain's Fairness Index - Wikipedia](https://en.wikipedia.org/wiki/Fairness_measure#Jain's_fairness_index)
- ExFairS论文中的 SAFI 定义
- 本项目的其他文档：`docs/InferSim_Complete_Guide.md`

