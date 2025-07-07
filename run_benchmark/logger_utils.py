#!/usr/bin/env python3
"""
日志工具模块
处理日志配置和设置
"""

import os
import logging
import glob
from datetime import datetime


def setup_logging():
    """设置日志记录器"""
    # 日志文件夹和文件名
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用全局配置中的时间戳
    from config.Config import GLOBAL_CONFIG
    timestamp = GLOBAL_CONFIG.get("monitor_file_time", "default")
    log_file = os.path.join(log_dir, f"run_benchmarks_{timestamp}.log")

    # 设置全局logger
    logger = logging.getLogger("run_benchmarks")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")  # 改回追加模式
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger 