# Ubuntu上编译和运行vLLM基准测试指南

## 🖥️ 系统要求

- **操作系统**: Ubuntu 18.04+ (推荐 Ubuntu 20.04/22.04)
- **GPU**: NVIDIA GPU (支持CUDA 11.8+)
- **内存**: 至少16GB RAM (推荐32GB+)
- **存储**: 至少50GB可用空间

## 📦 安装依赖

### 1. 更新系统包
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. 安装基础工具
```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    tree
```

### 3. 安装Python 3.9+
```bash
# 安装Python 3.9
sudo apt install -y python3.9 python3.9-dev python3.9-venv python3-pip

# 设置默认Python版本
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1
```

### 4. 安装CUDA (如果需要GPU支持)
```bash
# 下载CUDA 12.1 (或最新版本)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# 添加CUDA到PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 编译和安装vLLM

### 1. 克隆代码仓库
```bash
# 克隆你的基准测试项目
git clone <your-repo-url> vllm-benchmark
cd vllm-benchmark
```

### 2. 创建Python虚拟环境
```bash
# 创建虚拟环境
python3 -m venv vllm_env

# 激活虚拟环境
source vllm_env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### 3. 安装PyTorch
```bash
# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
```

### 4. 安装vLLM
```bash
# 方法1: 从PyPI安装 (推荐)
pip install vllm

# 方法2: 从源码编译 (如果需要最新功能)
# git clone https://github.com/vllm-project/vllm.git
# cd vllm
# pip install -e .
# cd ..
```

### 5. 安装其他依赖
```bash
# 安装基准测试所需的依赖
pip install -r requirements.txt

# 如果没有requirements.txt，手动安装常用依赖
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
    asyncio \
    psutil
```

## 📁 项目结构设置

### 1. 创建必要的目录
```bash
# 创建日志目录
mkdir -p log

# 创建模型目录
mkdir -p models

# 创建结果目录
mkdir -p results

# 创建prompt数据目录
mkdir -p prompt_hub
```

### 2. 下载测试数据
```bash
# 创建示例prompt文件
cat > prompt_hub/short_prompts.json << 'EOF'
[
    {"prompt": "What is the capital of France?", "max_tokens": 50},
    {"prompt": "Explain quantum computing in simple terms.", "max_tokens": 100},
    {"prompt": "Write a short poem about spring.", "max_tokens": 80}
]
EOF

cat > prompt_hub/long_prompts.json << 'EOF'
[
    {"prompt": "Write a detailed essay about the impact of artificial intelligence on modern society, covering both benefits and challenges.", "max_tokens": 500},
    {"prompt": "Explain the process of photosynthesis in plants, including the chemical reactions and environmental factors involved.", "max_tokens": 400}
]
EOF
```

## 🔧 配置文件设置

### 1. 创建配置文件
```bash
# 创建配置目录
mkdir -p config

# 创建基础配置文件
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
```

### 2. 创建requirements.txt
```bash
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
```

## 🏃‍♂️ 运行基准测试

### 1. 设置脚本权限
```bash
chmod +x start_vllm_benchmark.sh
```

### 2. 配置模型路径
```bash
# 编辑启动脚本，修改模型路径
vim start_vllm_benchmark.sh

# 修改以下行：
# MODEL_PATH="/path/to/your/model"  # 改为你的实际模型路径
# TOKENIZER_PATH="/path/to/your/tokenizer"  # 改为你的实际tokenizer路径
```

### 3. 运行基准测试
```bash
# 激活虚拟环境
source vllm_env/bin/activate

# 运行基准测试
./start_vllm_benchmark.sh
```

## 🐛 常见问题解决

### 1. CUDA相关问题
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 如果CUDA不可用，重新安装驱动
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535  # 或最新版本
sudo reboot
```

### 2. 内存不足问题
```bash
# 检查内存使用
free -h

# 增加交换空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Python包冲突
```bash
# 清理pip缓存
pip cache purge

# 重新创建虚拟环境
deactivate
rm -rf vllm_env
python3 -m venv vllm_env
source vllm_env/bin/activate
pip install --upgrade pip
```

### 4. 端口占用问题
```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 杀死占用端口的进程
sudo kill -9 <PID>
```

## 📊 性能优化建议

### 1. 系统优化
```bash
# 设置GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # 设置功率限制

# 优化系统性能
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. vLLM配置优化
```bash
# 在启动脚本中调整以下参数：
# - tensor_parallel_size: 根据GPU数量设置
# - max_num_seqs: 根据GPU内存调整
# - gpu_memory_utilization: 建议0.8-0.9
```

### 3. 监控工具
```bash
# 安装监控工具
pip install gpustat
sudo apt install iotop

# 实时监控GPU
watch -n 1 gpustat

# 监控系统资源
htop
```

## 🔄 自动化部署脚本

创建一键部署脚本：

```bash
cat > setup_ubuntu.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 开始Ubuntu环境设置..."

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3.9 python3.9-dev python3.9-venv python3-pip build-essential

# 创建虚拟环境
python3.9 -m venv vllm_env
source vllm_env/bin/activate

# 安装Python依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate

# 创建目录结构
mkdir -p log models results prompt_hub config

echo "✅ 环境设置完成！"
echo "请运行: source vllm_env/bin/activate 激活环境"
EOF

chmod +x setup_ubuntu.sh
```

## 📝 使用说明

1. **首次设置**: 运行 `./setup_ubuntu.sh` 进行一键环境配置
2. **日常使用**: 
   ```bash
   source vllm_env/bin/activate
   ./start_vllm_benchmark.sh
   ```
3. **查看结果**: 结果保存在 `results/` 目录下
4. **日志查看**: 日志保存在 `log/` 目录下

现在你可以在Ubuntu上顺利编译和运行vLLM基准测试了！🎉 