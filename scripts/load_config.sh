#!/bin/bash

# =============================================================================
# 配置加载脚本
# 从 YAML 文件加载场景和 vLLM 配置
# =============================================================================

SCENARIO_NAME="$1"
VLLM_CONFIG_FILE="$2"

# 加载 vLLM 配置
if [[ ! -f "$VLLM_CONFIG_FILE" ]]; then
    echo "❌ vLLM 配置文件不存在: $VLLM_CONFIG_FILE"
    exit 1
fi

VLLM_CONFIG_JSON=$(python3 -c "
import yaml, json
with open('$VLLM_CONFIG_FILE', 'r') as f:
    print(json.dumps(yaml.safe_load(f)))
")

# 导出 vLLM 参数
export MODEL_PATH=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")
export TOKENIZER_PATH=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['tokenizer_path'])")
export REQUEST_MODEL_NAME=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['request_model_name'])")
export TENSOR_PARALLEL_SIZE=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['tensor_parallel_size'])")
export GPU_MEMORY_UTILIZATION=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu_memory_utilization'])")
export MAX_MODEL_LEN=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['max_model_len'])")
export MAX_NUM_SEQS=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['max_num_seqs'])")
export MAX_NUM_BATCHED_TOKENS=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['max_num_batched_tokens'])")
export DTYPE=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['dtype'])")
export QUANTIZATION=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['quantization'])")
export DISABLE_LOG_STATS=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['disable_log_stats'])")
export ENABLE_PREFIX_CACHING=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['enable_prefix_caching'])")
export SCHEDULING_POLICY=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['scheduling_policy'])")
export START_ENGINE=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['start_engine'])")
export VLLM_URL=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['vllm_url'])")
export API_KEY=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['api_key'])")
export USE_TUNNEL=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['use_tunnel'])")
export LOCAL_PORT=$(echo "$VLLM_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['local_port'])")

# 加载场景配置
if [[ "$SCENARIO_NAME" != "default" ]]; then
    # 获取项目根目录（load_config.sh 在 scripts/ 目录下）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    # 场景文件在 config/scenarios/ 目录下
    SCENARIO_FILE="$PROJECT_ROOT/config/scenarios/${SCENARIO_NAME}.yaml"
    
    if [[ ! -f "$SCENARIO_FILE" ]]; then
        echo "❌ 场景配置文件不存在: $SCENARIO_FILE"
        exit 1
    fi
    
    # 同时加载 base_config.yaml 获取公共配置
    BASE_CONFIG_FILE="$PROJECT_ROOT/config/scenarios/base_config.yaml"
    
    # 使用 Python 合并 base_config 和场景配置
    SCENARIO_CONFIG_JSON=$(python3 -c "
import yaml, json
import os

# 加载 base_config
base_config = {}
if os.path.exists('$BASE_CONFIG_FILE'):
    with open('$BASE_CONFIG_FILE', 'r') as f:
        base_config = yaml.safe_load(f) or {}

# 加载场景配置
with open('$SCENARIO_FILE', 'r') as f:
    scenario_config = yaml.safe_load(f)

# 递归合并
def merge_dict(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result

merged = merge_dict(base_config, scenario_config)
print(json.dumps(merged))
")
    
    # 导出场景参数
    export SHORT_CLIENTS=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['short']['count'])")
    export SHORT_QPM=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['short']['qpm'])))")
    export SHORT_CLIENTS_SLO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['short']['slo'])))")
    export SHORT_CLIENT_QPM_RATIO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['short']['qpm_ratio'])")
    
    export LONG_CLIENTS=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['long']['count'])")
    export LONG_QPM=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['long']['qpm'])))")
    export LONG_CLIENTS_SLO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['long']['slo'])))")
    export LONG_CLIENT_QPM_RATIO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['long']['qpm_ratio'])")
    
    export MIX_CLIENTS=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['mix']['count'])")
    export MIX_QPM=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['mix']['qpm'])))")
    export MIX_CLIENTS_SLO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['clients']['mix']['slo'])))")
    export MIX_CLIENT_QPM_RATIO=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['clients']['mix']['qpm_ratio'])")
    
    export ROUND_NUM=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['round_num'])")
    export ROUND_TIME=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['round_time'])")
    export CONCURRENCY=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['concurrency'])")
    export NUM_REQUESTS=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['num_requests'])")
    export REQUEST_TIMEOUT=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['request_timeout'])")
    export SLEEP_TIME=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['sleep_time'])")
    export DISTRIBUTION=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['distribution'])")
    export USE_TIME_DATA=$(echo "$SCENARIO_CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['use_time_data'])")
else
    # 默认配置
    export SHORT_CLIENTS=0
    export SHORT_QPM=""
    export SHORT_CLIENTS_SLO=""
    export SHORT_CLIENT_QPM_RATIO=1
    export LONG_CLIENTS=0
    export LONG_QPM=""
    export LONG_CLIENTS_SLO=""
    export LONG_CLIENT_QPM_RATIO=1
    export MIX_CLIENTS=5
    export MIX_QPM="30 30 40 40 50"
    export MIX_CLIENTS_SLO="10 15 10 15 15"
    export MIX_CLIENT_QPM_RATIO=1
    export ROUND_NUM=10
    export ROUND_TIME=60
    export CONCURRENCY=1
    export NUM_REQUESTS=100
    export REQUEST_TIMEOUT=20
    export SLEEP_TIME=60
    export DISTRIBUTION="normal"
    export USE_TIME_DATA=0
fi

