"""
vLLM 引擎管理模块

提供直接启动和管理 vLLM 引擎的功能，无需预先启动 HTTP 服务器。

使用示例:
    from util.vllm import VLLMEngineManager
    
    # 创建引擎管理器（自动从 config/vllm/engine_config.yaml 加载配置）
    engine_manager = VLLMEngineManager()
    
    # 启动引擎
    engine = await engine_manager.start_engine()
    
    # 将引擎存储到全局配置
    from config.Config import GLOBAL_CONFIG
    GLOBAL_CONFIG['vllm_engine'] = engine
    
    # 关闭引擎
    await engine_manager.shutdown_engine()
"""

from util.vllm.engine_manager import (
    VLLMEngineManager, 
    setup_vllm_logging, 
    create_sampling_params,
    load_engine_config
)

__all__ = [
    'VLLMEngineManager', 
    'setup_vllm_logging', 
    'create_sampling_params',
    'load_engine_config'
]
