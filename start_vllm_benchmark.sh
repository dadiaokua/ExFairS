#!/bin/bash

# =============================================================================
# vLLM Benchmark 启动脚本
# 包含vLLM引擎参数配置和基准测试参数
# =============================================================================

# ========== 命令行参数解析 ==========
# 显示使用帮助
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                  显示此帮助信息"
    echo "  -e, --exp EXP_NAME          设置实验名称 (可多次使用，默认: QUEUE_ExFairS)"
    echo ""
    echo "可用的实验类型:"
    echo "  基础实验类型:"
    echo "  - ExFairS                   Experiential Fairness Scheduling"
    echo "  - VTC                       Variable Token Credits"
    echo "  - FCFS                      First Come First Serve"
    echo "  队列模式实验类型:"
    echo "  - QUEUE_ExFairS             队列模式 - Experiential Fairness调度"
    echo "  - QUEUE_VTC                 队列模式 - VTC调度"
    echo "  - QUEUE_FCFS                队列模式 - FCFS调度"
    echo "  - QUEUE_ROUND_ROBIN         队列模式 - 轮询调度"
    echo "  - QUEUE_MINQUE              队列模式 - QuE调度"
    echo ""
    echo "示例:"
    echo "  $0 -e ExFairS                  # 使用ExFairS实验类型"
    echo "  $0 -e QUEUE_VTC             # 使用队列模式VTC实验类型"
    echo "  $0 -e ExFairS -e VTC -e FCFS   # 依次运行三个实验类型"
    echo ""
}

# 实验类型映射函数：将用户输入的ExFairS映射为内部的LFS
map_experiment_type() {
    local input_exp="$1"
    case "$input_exp" in
        "ExFairS")
            echo "LFS"
            ;;
        "QUEUE_ExFairS")
            echo "QUEUE_LFS"
            ;;
        *)
            echo "$input_exp"
            ;;
    esac
}

# 初始化实验类型数组
EXP_NAMES=()

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--exp)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                EXP_NAMES+=("$2")
                shift 2
            else
                echo "错误: 参数 $1 需要一个值"
                show_help
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定实验类型，使用默认值
if [[ ${#EXP_NAMES[@]} -eq 0 ]]; then
    EXP_NAMES=("QUEUE_ExFairS")
fi

# ========== vLLM引擎参数 ==========
START_ENGINE="true"
MODEL_PATH="/home/llm/model_hub/Llama-3.1-8B"
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=8124
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=65536
SWAP_SPACE=0
DEVICE="cuda"
DTYPE="float16"
QUANTIZATION="None"
TRUST_REMOTE_CODE="true"
ENABLE_CHUNKED_PREFILL="false"
DISABLE_LOG_STATS="true"
ENABLE_PREFIX_CACHING="false"
SCHEDULING_POLICY="priority"

# ========== 基础连接参数 ==========
VLLM_URL="http://222.201.144.119:8000/v1"
API_KEY="None"
USE_TUNNEL=0
LOCAL_PORT="8000"
REMOTE_PORT="10085"

# ========== 请求配置参数 ==========
DISTRIBUTION="normal"
SHORT_QPM="10 90"
SHORT_CLIENT_QPM_RATIO=1
LONG_QPM="10 90"
LONG_CLIENT_QPM_RATIO=1
MIX_QPM="10 10 10 20 20 20 30 30"
MIX_CLIENT_QPM_RATIO=1

# ========== 客户端配置参数 ==========
SHORT_CLIENTS=0
SHORT_CLIENTS_SLO="20 20"
LONG_CLIENTS=0
LONG_CLIENTS_SLO="20 20"
MIX_CLIENTS=8
MIX_CLIENTS_SLO="10 10 15 15 20 20 20 25"

# ========== 并发和性能参数 ==========
CONCURRENCY=1
NUM_REQUESTS=100
REQUEST_TIMEOUT=20
SLEEP_TIME=60

# ========== 实验配置参数 ==========
ROUND_NUM=10
ROUND_TIME=60
USE_TIME_DATA=0

# ========== 模型和tokenizer参数 ==========
TOKENIZER_PATH="/home/llm/model_hub/Llama-3.1-8B"
REQUEST_MODEL_NAME="Qwen2.5-32B"

# ========== 验证实验类型 ==========
valid_exp_types=("ExFairS" "VTC" "FCFS" "QUEUE_ExFairS" "QUEUE_VTC" "QUEUE_FCFS" "QUEUE_ROUND_ROBIN" "QUEUE_MINQUE")
for exp_name in "${EXP_NAMES[@]}"; do
    if [[ ! " ${valid_exp_types[@]} " =~ " ${exp_name} " ]]; then
        echo "错误: 无效的实验类型 '$exp_name'"
        echo "支持的实验类型: ${valid_exp_types[*]}"
        exit 1
    fi
done

# ========== 显示配置信息 ==========
echo "=========================================="
echo "         vLLM Benchmark 配置信息"
echo "=========================================="
echo ""
echo "🧪 实验配置:"
echo "  将依次运行以下实验类型:"
for i in "${!EXP_NAMES[@]}"; do
    echo "    $((i+1)). ${EXP_NAMES[i]}"
done
echo "  总共 ${#EXP_NAMES[@]} 个实验"
echo ""
echo "🚀 vLLM引擎参数:"
echo "  Start Engine: $START_ENGINE"
echo "  Model Path: $MODEL_PATH"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Max Num Sequences: $MAX_NUM_SEQS"
echo "  Max Num Batched Tokens: $MAX_NUM_BATCHED_TOKENS"
echo "  Swap Space: ${SWAP_SPACE}GB"
echo "  Device: $DEVICE"
echo "  Data Type: $DTYPE"
echo "  Quantization: $QUANTIZATION"
echo "  Trust Remote Code: $TRUST_REMOTE_CODE"
echo "  Enable Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "  Disable Log Stats: $DISABLE_LOG_STATS"
echo "  Enable Prefix Caching: $ENABLE_PREFIX_CACHING"
echo "  Scheduling Policy: $SCHEDULING_POLICY"
echo ""
echo "🔗 基础连接参数:"
echo "  vLLM URL: $VLLM_URL"
echo "  API Key: $API_KEY"
echo "  Use Tunnel: $USE_TUNNEL"
echo "  Local Port: $LOCAL_PORT"
echo "  Remote Port: $REMOTE_PORT"
echo ""
echo "📊 请求配置参数:"
echo "  Distribution: $DISTRIBUTION"
echo "  Short QPM: $SHORT_QPM"
echo "  Short Client QPM Ratio: $SHORT_CLIENT_QPM_RATIO"
echo "  Long QPM: $LONG_QPM"
echo "  Long Client QPM Ratio: $LONG_CLIENT_QPM_RATIO"
echo "  MIX QPM: $MIX_QPM"
echo "  MIX Client QPM Ratio: $MIX_CLIENT_QPM_RATIO"
echo ""
echo "👥 客户端配置参数:"
echo "  Short Clients: $SHORT_CLIENTS"
echo "  Short Clients SLO: $SHORT_CLIENTS_SLO"
echo "  Long Clients: $LONG_CLIENTS"
echo "  Long Clients SLO: $LONG_CLIENTS_SLO"
echo "  MIX Clients: $MIX_CLIENTS"
echo "  MIX Clients SLO: $MIX_CLIENTS_SLO"
echo ""
echo "⚡ 并发和性能参数:"
echo "  Concurrency: $CONCURRENCY"
echo "  Num Requests: $NUM_REQUESTS"
echo "  Request Timeout: $REQUEST_TIMEOUT"
echo "  Sleep Time: $SLEEP_TIME"
echo ""
echo "🧪 实验配置参数:"
echo "  Round Num: $ROUND_NUM"
echo "  Round Time: $ROUND_TIME"
echo "  Use Time Data: $USE_TIME_DATA"
echo ""
echo "🤖 模型和tokenizer参数:"
echo "  Tokenizer Path: $TOKENIZER_PATH"
echo "  Request Model Name: $REQUEST_MODEL_NAME"
echo ""
echo "=========================================="
echo "正在启动基准测试..."
echo "=========================================="

# ========== 执行基准测试函数 ==========
run_benchmark() {
    local exp_name="$1"
    local exp_index="$2"
    local total_exps="$3"
    
    # 将实验类型映射为内部类型
    local internal_exp_name=$(map_experiment_type "$exp_name")
    
    echo ""
    echo "🚀🚀🚀 开始执行实验 $exp_index/$total_exps: $exp_name 🚀🚀🚀"
    echo "=========================================="
    
    cd run_benchmark
    
    python3 run_benchmarks.py \
        --vllm_url "$VLLM_URL" \
        --api_key "$API_KEY" \
        --use_tunnel "$USE_TUNNEL" \
        --local_port "$LOCAL_PORT" \
        --remote_port "$REMOTE_PORT" \
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
        --exp "$internal_exp_name" \
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
        --scheduling_policy "$SCHEDULING_POLICY"
    
    local exit_code=$?
    cd ..
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 实验 $exp_index/$total_exps: $exp_name 执行成功"
    else
        echo "❌ 实验 $exp_index/$total_exps: $exp_name 执行失败 (退出码: $exit_code)"
    fi
    
    echo "=========================================="
    
    return $exit_code
}

# ========== 启动所有基准测试 ==========
total_experiments=${#EXP_NAMES[@]}
failed_experiments=()
successful_experiments=()

for i in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[i]}"
    exp_index=$((i+1))
    
    if run_benchmark "$exp_name" "$exp_index" "$total_experiments"; then
        successful_experiments+=("$exp_name")
        echo "✅ 实验 $exp_index/$total_experiments: $exp_name 已完成"
    else
        failed_experiments+=("$exp_name")
        echo "❌ 实验 $exp_index/$total_experiments: $exp_name 执行失败"
    fi
    
    # 如果不是最后一个实验，显示即将开始下一个实验的提示
    if [ $exp_index -lt $total_experiments ]; then
        next_exp="${EXP_NAMES[i+1]}"
        echo ""
        echo "⏱️ 等待 180 秒后开始下一个实验..."
        sleep 180
        echo "📋 准备开始下一个实验: $((exp_index+1))/$total_experiments - $next_exp"
        echo "=========================================="
    fi
done

# ========== 显示最终结果 ==========
echo ""
echo "🎉🎉🎉 所有实验执行完成！🎉🎉🎉"
echo "=========================================="
echo "📊 执行结果总览:"
echo "  总实验数: $total_experiments"
echo "  成功实验数: ${#successful_experiments[@]}"
echo "  失败实验数: ${#failed_experiments[@]}"
echo ""

if [ ${#successful_experiments[@]} -gt 0 ]; then
    echo "✅ 成功的实验:"
    for exp in "${successful_experiments[@]}"; do
        echo "    - $exp"
    done
    echo ""
fi

if [ ${#failed_experiments[@]} -gt 0 ]; then
    echo "❌ 失败的实验:"
    for exp in "${failed_experiments[@]}"; do
        echo "    - $exp"
    done
    echo ""
    echo "⚠️ 请检查失败实验的日志以了解详细错误信息"
fi

echo "=========================================="
echo "基准测试批处理完成！"
echo "=========================================="

# 如果有失败的实验，以非零状态码退出
if [ ${#failed_experiments[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
