#!/bin/bash

# =============================================================================
# vLLM Benchmark 主运行脚本 - 实时监控模式
# =============================================================================
# 核心特点：
# 1. 只做一轮，持续时间长（可配置，默认10分钟）
# 2. 后台监控器每60秒收集一次数据
# 3. 实时计算SAFI和优先级交换
# 4. 不阻塞推理，更新后的优先级立即作用于新请求
# =============================================================================

set -e

# ========== 路径配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE_DIR="$SCRIPT_DIR/results"

# ========== 默认配置 ==========
DEFAULT_DURATION=600        # 默认10分钟
DEFAULT_INTERVAL=60         # 默认60秒监控间隔
DEFAULT_PORT=8000
DEFAULT_MODEL=""
DEFAULT_DATASET="sharegpt"

# ========== 帮助信息 ==========
show_help() {
    cat << EOF
使用方法: $0 [选项]

基本选项:
  -h, --help                    显示此帮助信息
  -e, --exp EXP1,EXP2,...       指定实验策略（逗号分隔）
  -s, --scenarios S1,S2,...     指定场景（逗号分隔）
  --scenario SCENARIO           指定单个场景
  --duration SECONDS            实验持续时间（秒，默认600=10分钟）
  --interval SECONDS            监控间隔（秒，默认60）
  --port PORT                   vLLM服务端口（默认8000）
  --model MODEL                 模型路径/名称（用于tokenizer）
  --dataset DATASET             数据集名称（默认sharegpt）
  --output-dir DIR              指定输出目录

可用的调度策略:
  - QUEUE_ExFairS               体验式公平调度（我们的方法）
  - QUEUE_Justitia              虚拟时间调度
  - QUEUE_SLOGreedy             SLO违约率贪心调度
  - QUEUE_VTC                   可变Token积分
  - QUEUE_FCFS                  先到先服务
  - QUEUE_RR                    轮询调度

可用场景 (支持数字或完整名称):
  - 1 / scenario_I              Mix2: QPM不均 (20,60), SLO统一 (20s)
  - 2 / scenario_II             Mix2: QPM均匀 (40,40), SLO不均 (15,20s)
  - 3 / scenario_III            Mix4: QPM递增 (15-30), SLO统一 (15s)
  - 4 / scenario_IV             Mix4: QPM均匀 (20), SLO递增 (10-20s)
  - 5 / scenario_V              Mix6: 综合差异场景

示例:
  # 默认运行（场景1 + ExFairS）
  $0
  
  # 单场景单策略
  $0 -e QUEUE_ExFairS --scenario 1
  
  # 单场景多策略
  $0 -e QUEUE_ExFairS,QUEUE_VTC,QUEUE_FCFS --scenario 1
  
  # 多场景多策略（批量运行）
  $0 -e QUEUE_ExFairS,QUEUE_VTC -s 1,2,3
  
  # 自定义持续时间和监控间隔
  $0 -e QUEUE_ExFairS --scenario 1 --duration 300 --interval 30

  # 运行所有场景所有策略
  $0 -s 1,2,3,4,5 -e QUEUE_ExFairS,QUEUE_Justitia,QUEUE_SLOGreedy,QUEUE_VTC,QUEUE_FCFS

查询选项:
  $0 --list-scenarios           列出所有可用场景
  $0 --list-strategies          列出所有可用策略

EOF
}

# ========== 场景数字映射 ==========
map_scenario() {
    local input="$1"
    case "$input" in
        1) echo "scenario_I" ;;
        2) echo "scenario_II" ;;
        3) echo "scenario_III" ;;
        4) echo "scenario_IV" ;;
        5) echo "scenario_V" ;;
        scenario_*) echo "$input" ;;
        *) echo "$input" ;;
    esac
}

map_scenarios() {
    local input="$1"
    local result=""
    IFS=',' read -ra arr <<< "$input"
    for s in "${arr[@]}"; do
        mapped=$(map_scenario "$s")
        [[ -n "$result" ]] && result+=","
        result+="$mapped"
    done
    echo "$result"
}

list_scenarios() {
    echo "可用场景:"
    echo "  1 - scenario_I:   Mix2: QPM不均 (20,60), SLO统一 (20s)"
    echo "  2 - scenario_II:  Mix2: QPM均匀 (40,40), SLO不均 (15,20s)"
    echo "  3 - scenario_III: Mix4: QPM递增 (15-30), SLO统一 (15s)"
    echo "  4 - scenario_IV:  Mix4: QPM均匀 (20), SLO递增 (10-20s)"
    echo "  5 - scenario_V:   Mix6: 综合差异场景"
}

list_strategies() {
    cat << EOF
可用调度策略:
  - QUEUE_ExFairS               体验式公平调度（我们的方法）
  - QUEUE_Justitia              虚拟时间调度
  - QUEUE_SLOGreedy             SLO违约率贪心调度
  - QUEUE_VTC                   可变Token积分
  - QUEUE_FCFS                  先到先服务
  - QUEUE_RR                    轮询调度
EOF
}

# ========== 格式化时间函数 ==========
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# ========== 参数解析 ==========
EXPERIMENTS=""
SCENARIOS=""
SINGLE_SCENARIO=""
DURATION=$DEFAULT_DURATION
INTERVAL=$DEFAULT_INTERVAL
PORT=$DEFAULT_PORT
MODEL=$DEFAULT_MODEL
DATASET=$DEFAULT_DATASET
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --list-scenarios)
            list_scenarios
            exit 0
            ;;
        --list-strategies)
            list_strategies
            exit 0
            ;;
        -e|--exp)
            EXPERIMENTS="$2"
            shift 2
            ;;
        -s|--scenarios)
            SCENARIOS="$2"
            shift 2
            ;;
        --scenario)
            SINGLE_SCENARIO="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# ========== 确定场景列表 ==========
# 如果指定了单场景
[[ -n "$SINGLE_SCENARIO" ]] && SINGLE_SCENARIO=$(map_scenario "$SINGLE_SCENARIO")
[[ -n "$SCENARIOS" ]] && SCENARIOS=$(map_scenarios "$SCENARIOS")

# 默认值
[[ -z "$SCENARIOS" && -z "$SINGLE_SCENARIO" ]] && SINGLE_SCENARIO="scenario_I"
[[ -z "$EXPERIMENTS" ]] && EXPERIMENTS="QUEUE_ExFairS"

# 如果只有单场景，转换为场景列表
[[ -n "$SINGLE_SCENARIO" && -z "$SCENARIOS" ]] && SCENARIOS="$SINGLE_SCENARIO"

# 转换为数组
IFS=',' read -ra SCENARIO_ARRAY <<< "$SCENARIOS"
IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"

# ========== 生成运行ID ==========
RUN_TIMESTAMP="run_$(date +"%Y%m%d_%H%M%S")"
RUN_RESULTS_DIR="${OUTPUT_DIR:-$RESULTS_BASE_DIR/$RUN_TIMESTAMP}"
mkdir -p "$RUN_RESULTS_DIR"

LOG_FILE="$RUN_RESULTS_DIR/run.log"

# ========== 打印运行配置 ==========
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          vLLM Benchmark - 实时监控模式                         ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  场景: ${SCENARIO_ARRAY[@]}"
echo "║  策略: ${EXP_ARRAY[@]}"
echo "║  持续时间: ${DURATION}s ($(format_duration $DURATION))"
echo "║  监控间隔: ${INTERVAL}s"
echo "║  vLLM端口: $PORT"
echo "║  结果目录: $RUN_RESULTS_DIR"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 记录到日志
{
    echo "=========================================="
    echo "vLLM Benchmark - 实时监控模式"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "场景: ${SCENARIO_ARRAY[@]}"
    echo "策略: ${EXP_ARRAY[@]}"
    echo "持续时间: ${DURATION}s"
    echo "监控间隔: ${INTERVAL}s"
    echo "vLLM端口: $PORT"
    echo "=========================================="
} >> "$LOG_FILE"

# ========== 运行统计 ==========
total_runs=$((${#SCENARIO_ARRAY[@]} * ${#EXP_ARRAY[@]}))
run_counter=0
success_counter=0
failed_runs=()

BATCH_START_TIME=$(date +%s)

# ========== 批量运行 ==========
for scenario in "${SCENARIO_ARRAY[@]}"; do
    SCENARIO_START_TIME=$(date +%s)
    
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    echo "║  开始场景: $scenario" | tee -a "$LOG_FILE"
    echo "║  开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════╝" | tee -a "$LOG_FILE"
    
    scenario_success=0
    
    for exp in "${EXP_ARRAY[@]}"; do
        run_counter=$((run_counter + 1))
        EXP_START_TIME=$(date +%s)
        
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "🚀 运行 $run_counter/$total_runs: $scenario + $exp" | tee -a "$LOG_FILE"
        echo "⏰ 开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        
        # 构建Python命令参数
        CMD_ARGS="--scenario $scenario --strategy $exp --duration $DURATION --interval $INTERVAL --port $PORT --run-id $RUN_TIMESTAMP"
        [[ -n "$MODEL" ]] && CMD_ARGS="$CMD_ARGS --model $MODEL"
        [[ -n "$DATASET" ]] && CMD_ARGS="$CMD_ARGS --dataset $DATASET"
        
        # 运行实时监控实验
        cd "$SCRIPT_DIR"
        if python3 scripts/run_realtime_benchmark.py $CMD_ARGS 2>&1 | tee -a "$LOG_FILE"; then
            success_counter=$((success_counter + 1))
            scenario_success=$((scenario_success + 1))
            EXP_END_TIME=$(date +%s)
            EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
            echo "✅ 完成 $run_counter/$total_runs (耗时: $(format_duration $EXP_DURATION))" | tee -a "$LOG_FILE"
        else
            EXP_END_TIME=$(date +%s)
            EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
            echo "❌ 失败 $run_counter/$total_runs (耗时: $(format_duration $EXP_DURATION))" | tee -a "$LOG_FILE"
            failed_runs+=("$scenario + $exp")
        fi
        
        # 策略之间等待
        exp_index=0
        for i in "${!EXP_ARRAY[@]}"; do
            [[ "${EXP_ARRAY[$i]}" == "$exp" ]] && exp_index=$i && break
        done
        if [[ $((exp_index + 1)) -lt ${#EXP_ARRAY[@]} ]]; then
            echo "⏱️  等待 5 秒..." | tee -a "$LOG_FILE"
            sleep 5
        fi
    done
    
    # 场景完成，生成可视化
    echo "" | tee -a "$LOG_FILE"
    echo "📊 场景 $scenario 完成，生成可视化..." | tee -a "$LOG_FILE"
    
    if [[ -f "$SCRIPT_DIR/scripts/visualize_results.py" ]]; then
        cd "$SCRIPT_DIR"
        if python3 scripts/visualize_results.py "$scenario" --run-id "$RUN_TIMESTAMP" --results-dir "$RESULTS_BASE_DIR" >> "$LOG_FILE" 2>&1; then
            echo "✅ 可视化完成: $RUN_RESULTS_DIR/$scenario/charts/" | tee -a "$LOG_FILE"
        else
            echo "⚠️  可视化失败（不影响实验结果）" | tee -a "$LOG_FILE"
        fi
    fi
    
    SCENARIO_END_TIME=$(date +%s)
    SCENARIO_DURATION=$((SCENARIO_END_TIME - SCENARIO_START_TIME))
    
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    echo "║  ✨ 场景 $scenario 完成" | tee -a "$LOG_FILE"
    echo "║  成功: $scenario_success/${#EXP_ARRAY[@]}" | tee -a "$LOG_FILE"
    echo "║  场景耗时: $(format_duration $SCENARIO_DURATION)" | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
    
    # 场景之间等待
    scenario_index=0
    for i in "${!SCENARIO_ARRAY[@]}"; do
        [[ "${SCENARIO_ARRAY[$i]}" == "$scenario" ]] && scenario_index=$i && break
    done
    if [[ $((scenario_index + 1)) -lt ${#SCENARIO_ARRAY[@]} ]]; then
        echo "⏸️  场景间隔，等待 5 秒..." | tee -a "$LOG_FILE"
        sleep 5
    fi
done

# ========== 总结 ==========
BATCH_END_TIME=$(date +%s)
TOTAL_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "╔════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║                       🎉 全部运行完成                           ║" | tee -a "$LOG_FILE"
echo "╠════════════════════════════════════════════════════════════════╣" | tee -a "$LOG_FILE"
echo "║  总运行数: $total_runs" | tee -a "$LOG_FILE"
echo "║  成功: $success_counter" | tee -a "$LOG_FILE"
echo "║  失败: $((total_runs - success_counter))" | tee -a "$LOG_FILE"
echo "╠════════════════════════════════════════════════════════════════╣" | tee -a "$LOG_FILE"
echo "║  总耗时: $(format_duration $TOTAL_DURATION)" | tee -a "$LOG_FILE"
echo "║  开始时间: $(date -d @$BATCH_START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $BATCH_START_TIME '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "║  结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "║  结果目录: $RUN_RESULTS_DIR" | tee -a "$LOG_FILE"
echo "╚════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

# 保存元数据
cat > "$RUN_RESULTS_DIR/metadata.json" << EOF
{
  "run_timestamp": "$RUN_TIMESTAMP",
  "mode": "realtime",
  "scenarios": [$(printf '"%s",' "${SCENARIO_ARRAY[@]}" | sed 's/,$//')],
  "experiments": [$(printf '"%s",' "${EXP_ARRAY[@]}" | sed 's/,$//')],
  "duration_seconds": $DURATION,
  "interval_seconds": $INTERVAL,
  "port": $PORT,
  "total_runs": $total_runs,
  "successful_runs": $success_counter,
  "failed_runs": $((total_runs - success_counter)),
  "start_time": "$(date -d @$BATCH_START_TIME '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -r $BATCH_START_TIME '+%Y-%m-%dT%H:%M:%S')",
  "end_time": "$(date '+%Y-%m-%dT%H:%M:%S')",
  "total_duration_seconds": $TOTAL_DURATION,
  "total_duration_formatted": "$(format_duration $TOTAL_DURATION)"
}
EOF

# 打印失败的运行
if [[ ${#failed_runs[@]} -gt 0 ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "❌ 失败的运行:" | tee -a "$LOG_FILE"
    for run in "${failed_runs[@]}"; do
        echo "  - $run" | tee -a "$LOG_FILE"
    done
fi

exit $([[ ${#failed_runs[@]} -eq 0 ]] && echo 0 || echo 1)
