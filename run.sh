#!/bin/bash

# =============================================================================
# vLLM Benchmark 主运行脚本
# 支持单场景运行和批量多场景运行
# =============================================================================

set -e

# ========== 路径配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
RESULTS_BASE_DIR="$SCRIPT_DIR/results"

# ========== 帮助信息 ==========
show_help() {
    cat << EOF
使用方法: $0 [选项]

基本选项:
  -h, --help                    显示此帮助信息
  -e, --exp EXP1,EXP2,...       指定实验策略（逗号分隔）
  -s, --scenarios S1,S2,...     指定场景（逗号分隔，批量模式）
  --scenario SCENARIO           指定单个场景（单场景模式）
  --output-dir DIR              指定输出目录
  --vllm-config FILE            指定vLLM配置文件

模式:
  单场景模式: 使用 --scenario 指定一个场景
  批量模式: 使用 -s 指定多个场景（自动批量运行）

可用的实验策略:
  - QUEUE_ExFairS              队列模式 - ExFairS调度
  - QUEUE_Justitia             队列模式 - Justitia虚拟时间调度
  - QUEUE_SLOGreedy            队列模式 - SLO违约率贪心调度
  - QUEUE_VTC                  队列模式 - VTC调度
  - QUEUE_FCFS                 队列模式 - FCFS调度
  - QUEUE_ROUND_ROBIN          队列模式 - 轮询调度
  - QUEUE_MINQUE               队列模式 - QuE调度
  - ExFairS, Justitia, SLOGreedy, VTC, FCFS  (基础模式)

可用场景:
  - scenario_I                 均衡负载 (2S+2L, QPM=50)
  - scenario_II                不均衡负载 (2S+2L, QPM=10-90)
  - scenario_III               异构4客户端 (Mix, QPM=20-40)
  - scenario_IV                异构8客户端 (Mix, QPM=10-30)
  - scenario_V                 高并发20客户端 (Mix, QPM=5-15)
  - scenario_VI                高并发50客户端 (Mix, QPM=4)

示例:
  # 单场景运行
  $0 -e QUEUE_ExFairS --scenario scenario_I
  
  # 单场景多策略
  $0 -e QUEUE_ExFairS,QUEUE_Justitia --scenario scenario_I
  
  # 批量运行多场景多策略
  $0 -e QUEUE_ExFairS,QUEUE_Justitia -s scenario_I,scenario_II
  
  # 运行所有场景和默认策略
  $0 -s scenario_I,scenario_II,scenario_III,scenario_IV,scenario_V,scenario_VI

查询选项:
  $0 --list-scenarios          列出所有可用场景
  $0 --list-strategies         列出所有可用策略

EOF
}

list_scenarios() {
    echo "可用场景:"
    python3 "$CONFIG_DIR/scenario_manager.py" list 2>/dev/null || {
        echo "  - scenario_I: 均衡负载"
        echo "  - scenario_II: 不均衡负载"
        echo "  - scenario_III: 异构4客户端"
        echo "  - scenario_IV: 异构8客户端"
        echo "  - scenario_V: 高并发20客户端"
        echo "  - scenario_VI: 高并发50客户端"
    }
}

list_strategies() {
    cat << EOF
可用调度策略:
  队列模式 (推荐):
    - QUEUE_ExFairS              体验式公平调度
    - QUEUE_Justitia             虚拟时间调度
    - QUEUE_SLOGreedy            SLO违约率贪心
    - QUEUE_VTC                  可变Token积分
    - QUEUE_FCFS                 先到先服务
    - QUEUE_ROUND_ROBIN          轮询调度
    - QUEUE_MINQUE               QuE调度

  基础模式:
    - ExFairS, Justitia, SLOGreedy, VTC, FCFS
EOF
}

# ========== 参数解析 ==========
EXPERIMENTS=""
SCENARIOS=""
SINGLE_SCENARIO=""
OUTPUT_DIR=""
VLLM_CONFIG="$CONFIG_DIR/vllm/engine_config.yaml"
BATCH_MODE=false

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
            BATCH_MODE=true
            shift 2
            ;;
        --scenario)
            SINGLE_SCENARIO="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vllm-config)
            VLLM_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# ========== 模式判断 ==========
if [[ "$BATCH_MODE" == true ]]; then
    # 批量模式
    echo "🚀 批量运行模式"
    
    # 默认值
    [[ -z "$SCENARIOS" ]] && SCENARIOS="scenario_I,scenario_II,scenario_III,scenario_IV,scenario_V,scenario_VI"
    [[ -z "$EXPERIMENTS" ]] && EXPERIMENTS="QUEUE_ExFairS,QUEUE_Justitia,QUEUE_SLOGreedy,QUEUE_VTC,QUEUE_FCFS"
    
    # 转换为数组
    IFS=',' read -ra SCENARIO_ARRAY <<< "$SCENARIOS"
    IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
    
    # 生成时间戳
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_RESULTS_DIR="${OUTPUT_DIR:-$RESULTS_BASE_DIR/$RUN_TIMESTAMP}"
    mkdir -p "$RUN_RESULTS_DIR"
    
    LOG_FILE="$RUN_RESULTS_DIR/run.log"
    
    echo "场景: ${SCENARIO_ARRAY[@]}"
    echo "策略: ${EXP_ARRAY[@]}"
    echo "结果目录: $RUN_RESULTS_DIR"
    echo ""
    
    total_runs=$((${#SCENARIO_ARRAY[@]} * ${#EXP_ARRAY[@]}))
    run_counter=0
    success_counter=0
    failed_runs=()
    
    # 批量运行
    for scenario in "${SCENARIO_ARRAY[@]}"; do
        for exp in "${EXP_ARRAY[@]}"; do
            run_counter=$((run_counter + 1))
            echo "========================================" | tee -a "$LOG_FILE"
            echo "🚀 运行 $run_counter/$total_runs: $scenario + $exp" | tee -a "$LOG_FILE"
            echo "========================================" | tee -a "$LOG_FILE"
            
            run_dir="$RUN_RESULTS_DIR/${scenario}_${exp}"
            mkdir -p "$run_dir"
            
            # 调用单场景运行逻辑
            if $0 -e "$exp" --scenario "$scenario" --output-dir "$run_dir" >> "$LOG_FILE" 2>&1; then
                success_counter=$((success_counter + 1))
                echo "✅ 完成 $run_counter/$total_runs" | tee -a "$LOG_FILE"
            else
                echo "❌ 失败 $run_counter/$total_runs" | tee -a "$LOG_FILE"
                failed_runs+=("$scenario + $exp")
            fi
            
            # 等待
            if [[ $run_counter -lt $total_runs ]]; then
                echo "⏱️  等待 30 秒..." | tee -a "$LOG_FILE"
                sleep 30
            fi
        done
    done
    
    # 总结
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "🎉 批量运行完成" | tee -a "$LOG_FILE"
    echo "总运行: $total_runs" | tee -a "$LOG_FILE"
    echo "成功: $success_counter" | tee -a "$LOG_FILE"
    echo "失败: $((total_runs - success_counter))" | tee -a "$LOG_FILE"
    echo "结果: $RUN_RESULTS_DIR" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 保存元数据
    cat > "$RUN_RESULTS_DIR/metadata.json" << EOF
{
  "run_timestamp": "$RUN_TIMESTAMP",
  "scenarios": [$(printf '"%s",' "${SCENARIO_ARRAY[@]}" | sed 's/,$//')],
  "experiments": [$(printf '"%s",' "${EXP_ARRAY[@]}" | sed 's/,$//')],
  "total_runs": $total_runs,
  "successful_runs": $success_counter,
  "failed_runs": $((total_runs - success_counter))
}
EOF
    
    exit $([[ ${#failed_runs[@]} -eq 0 ]] && echo 0 || echo 1)
    
else
    # 单场景模式
    echo "🎯 单场景运行模式"
    
    [[ -z "$SINGLE_SCENARIO" ]] && SINGLE_SCENARIO="default"
    [[ -z "$EXPERIMENTS" ]] && EXPERIMENTS="QUEUE_ExFairS"
    [[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$RESULTS_BASE_DIR/$(date +"%Y%m%d_%H%M%S")"
    
    mkdir -p "$OUTPUT_DIR"
    
    IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
    
    echo "场景: $SINGLE_SCENARIO"
    echo "策略: ${EXP_ARRAY[@]}"
    echo "输出: $OUTPUT_DIR"
    echo ""
    
    # 加载配置
    source "$SCRIPT_DIR/scripts/load_config.sh" "$SINGLE_SCENARIO" "$VLLM_CONFIG"
    
    # 运行实验
    success_count=0
    for exp in "${EXP_ARRAY[@]}"; do
        echo "========================================" 
        echo "🚀 运行实验: $exp"
        echo "========================================" 
        
        # 映射实验类型
        internal_exp="$exp"
        [[ "$exp" == "ExFairS" ]] && internal_exp="LFS"
        [[ "$exp" == "QUEUE_ExFairS" ]] && internal_exp="QUEUE_LFS"
        
        cd "$SCRIPT_DIR/run_benchmark"
        
        if python3 run_benchmarks.py \
            --vllm_url "$VLLM_URL" \
            --api_key "$API_KEY" \
            --use_tunnel "$USE_TUNNEL" \
            --local_port "$LOCAL_PORT" \
            --distribution "$DISTRIBUTION" \
            --short_qpm "$SHORT_QPM" \
            --short_client_qpm_ratio "$SHORT_CLIENT_QPM_RATIO" \
            --long_qpm "$LONG_QPM" \
            --long_client_qpm_ratio "$LONG_CLIENT_QPM_RATIO" \
            --mix_qpm "$MIX_QPM" \
            --mix_client_qpm_ratio "$MIX_CLIENT_QPM_RATIO" \
            --short_clients "$SHORT_CLIENTS" \
            --short_clients_slo "$SHORT_CLIENTS_SLO" \
            --long_clients "$LONG_CLIENTS" \
            --long_clients_slo "$LONG_CLIENTS_SLO" \
            --mix_clients "$MIX_CLIENTS" \
            --mix_clients_slo "$MIX_CLIENTS_SLO" \
            --concurrency "$CONCURRENCY" \
            --num_requests "$NUM_REQUESTS" \
            --request_timeout "$REQUEST_TIMEOUT" \
            --sleep "$SLEEP_TIME" \
            --round "$ROUND_NUM" \
            --round_time "$ROUND_TIME" \
            --exp "$internal_exp" \
            --scenario "$SINGLE_SCENARIO" \
            --use_time_data "$USE_TIME_DATA" \
            --tokenizer "$TOKENIZER_PATH" \
            --request_model_name "$REQUEST_MODEL_NAME" \
            --start_engine "$START_ENGINE" \
            --model_path "$MODEL_PATH" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$MAX_NUM_SEQS" \
            --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
            --dtype "$DTYPE" \
            --quantization "$QUANTIZATION" \
            --disable_log_stats "$DISABLE_LOG_STATS" \
            --enable_prefix_caching "$ENABLE_PREFIX_CACHING" \
            --scheduling_policy "$SCHEDULING_POLICY"; then
            
            success_count=$((success_count + 1))
            echo "✅ $exp 完成"
            
            # 复制结果
            cp -r "$SCRIPT_DIR/results/"* "$OUTPUT_DIR/" 2>/dev/null || true
            cp -r "$SCRIPT_DIR/figure/"* "$OUTPUT_DIR/" 2>/dev/null || true
        else
            echo "❌ $exp 失败"
        fi
        
        cd "$SCRIPT_DIR"
        
        # 等待
        if [[ ${#EXP_ARRAY[@]} -gt 1 ]]; then
            echo "⏱️  等待 30 秒..."
            sleep 30
        fi
    done
    
    echo ""
    echo "========================================" 
    echo "🎉 运行完成"
    echo "成功: $success_count / ${#EXP_ARRAY[@]}"
    echo "结果: $OUTPUT_DIR"
    echo "========================================" 
    
    exit $([[ $success_count -eq ${#EXP_ARRAY[@]} ]] && echo 0 || echo 1)
fi

