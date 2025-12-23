#!/bin/bash

# =============================================================================
# Ubuntu一键部署脚本 - vLLM基准测试环境
# =============================================================================

set -e  # 遇到错误立即退出

echo "🚀 开始Ubuntu环境设置..."
echo "=========================================="

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then
    echo "❌ 请不要以root用户运行此脚本"
    exit 1
fi

# ========== 1. 更新系统 ==========
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# ========== 2. 安装基础工具 ==========
echo "🔧 安装基础工具..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    tree \
    unzip \
    software-properties-common

# ========== 3. 安装Python 3.9+ ==========
echo "🐍 安装Python 3.9..."
sudo apt install -y python3.9 python3.9-dev python3.9-venv python3-pip

# 设置Python版本
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# ========== 4. 检查NVIDIA GPU ==========
echo "🖥️ 检查GPU支持..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ 未检测到NVIDIA GPU或驱动，将安装CPU版本"
fi

# ========== 5. 创建虚拟环境 ==========
echo "🔧 创建Python虚拟环境..."
python3.9 -m venv vllm_env

# 激活虚拟环境
source vllm_env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# ========== 6. 安装PyTorch ==========
echo "🔥 安装PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # 安装CUDA版本
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "✅ 安装CUDA版本PyTorch"
else
    # 安装CPU版本
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "✅ 安装CPU版本PyTorch"
fi

# 验证PyTorch安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
if command -v nvidia-smi &> /dev/null; then
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"
fi

# ========== 7. 安装vLLM ==========
echo "🚀 安装vLLM..."
pip install vllm

# ========== 8. 安装其他依赖 ==========
echo "📚 安装其他依赖..."
pip install \
    transformers \
    accelerate \
    datasets \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    requests \
    openai \
    aiohttp \
    psutil \
    pyyaml

# ========== 9. 创建项目目录结构 ==========
echo "📁 创建项目目录结构..."
mkdir -p log
mkdir -p models
mkdir -p results
mkdir -p prompt_hub
mkdir -p config

# ========== 10. 创建示例prompt文件 ==========
echo "📝 创建示例prompt文件..."
cat > prompt_hub/short_prompts.json << 'EOF'
[
    {"prompt": "What is the capital of France?", "max_tokens": 50},
    {"prompt": "Explain quantum computing in simple terms.", "max_tokens": 100},
    {"prompt": "Write a short poem about spring.", "max_tokens": 80},
    {"prompt": "How does photosynthesis work?", "max_tokens": 120},
    {"prompt": "What are the benefits of renewable energy?", "max_tokens": 90}
]
EOF

cat > prompt_hub/long_prompts.json << 'EOF'
[
    {"prompt": "Write a detailed essay about the impact of artificial intelligence on modern society, covering both benefits and challenges.", "max_tokens": 500},
    {"prompt": "Explain the process of photosynthesis in plants, including the chemical reactions and environmental factors involved.", "max_tokens": 400},
    {"prompt": "Describe the history and development of the internet, from its origins to modern day applications.", "max_tokens": 600}
]
EOF

# ========== 11. 创建配置文件 ==========
echo "⚙️ 创建配置文件..."
cat > config/Config.py << 'EOF'
#!/usr/bin/env python3
"""
全局配置文件
"""

GLOBAL_CONFIG = {
    'exp_time': 36000,  # 实验超时时间(秒)
    'round_time': 600,  # 每轮超时时间(秒)
    'request_model_name': 'default-model',  # 默认模型名称
    'vllm_engine': None,  # vLLM引擎实例
}

# 结果文件路径
RESULTS_FILE = "../results/benchmark_results.json"
EOF

# ========== 12. 创建requirements.txt ==========
echo "📋 创建requirements.txt..."
cat > requirements.txt << 'EOF'
# vLLM和相关依赖
vllm>=0.2.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# 数据处理
numpy>=1.21.0
pandas>=1.5.0
datasets>=2.0.0

# 网络请求
requests>=2.28.0
aiohttp>=3.8.0
openai>=0.27.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0

# 工具库
tqdm>=4.64.0
psutil>=5.9.0
pyyaml>=6.0
EOF

# ========== 13. 设置脚本权限 ==========
echo "🔐 设置脚本权限..."
chmod +x start_vllm_benchmark.sh

# ========== 14. 创建快速启动脚本 ==========
echo "🎯 创建快速启动脚本..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 启动vLLM基准测试..."
source vllm_env/bin/activate
./start_vllm_benchmark.sh
EOF
chmod +x quick_start.sh

# ========== 15. 安装监控工具 ==========
echo "📊 安装监控工具..."
pip install gpustat
sudo apt install -y iotop htop

# ========== 完成安装 ==========
echo ""
echo "=========================================="
echo "🎉 Ubuntu环境设置完成！"
echo "=========================================="
echo ""
echo "📁 项目结构:"
echo "  ├── vllm_env/              # Python虚拟环境"
echo "  ├── run_benchmark/         # 基准测试代码"
echo "  ├── prompt_hub/            # 测试prompt数据"
echo "  ├── config/                # 配置文件"
echo "  ├── log/                   # 日志文件"
echo "  ├── results/               # 结果文件"
echo "  ├── models/                # 模型文件"
echo "  └── requirements.txt       # 依赖列表"
echo ""
echo "🚀 使用方法:"
echo "  1. 激活环境: source vllm_env/bin/activate"
echo "  2. 运行测试: ./start_vllm_benchmark.sh"
echo "  3. 或者使用: ./quick_start.sh (自动激活环境)"
echo ""
echo "📊 监控工具:"
echo "  • GPU监控: watch -n 1 gpustat"
echo "  • 系统监控: htop"
echo "  • IO监控: sudo iotop"
echo ""
echo "⚠️ 注意事项:"
echo "  • 请确保模型路径正确配置"
echo "  • 根据GPU内存调整batch size"
echo "  • 查看日志: tail -f log/run_benchmarks.log"
echo ""
echo "✅ 环境准备就绪，可以开始基准测试了！" 