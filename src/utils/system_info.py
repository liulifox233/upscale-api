"""
系统信息模块：获取和打印系统信息
"""
import platform
import psutil
import sys
import torch
import logging
from typing import Dict, Any, List

logger = logging.getLogger("upscale-api")

def get_system_info() -> Dict[str, Any]:
    """收集系统信息"""
    info = {
        "os": f"{platform.system()} {platform.version()}",
        "python_version": sys.version.split()[0],
        "cpu": platform.processor() or "Unknown",
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
        "torch_version": torch.__version__,
        "gpu_available": "No"
    }
    
    # 获取GPU信息
    if torch.cuda.is_available():
        info["gpu_available"] = "CUDA"
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_info"] = []
        
        for i in range(torch.cuda.device_count()):
            gpu = {
                "name": torch.cuda.get_device_name(i),
                "memory": f"{round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)} GB"
            }
            info["gpu_info"].append(gpu)
    
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        info["gpu_available"] = "MPS (Apple Metal)"
    
    return info

def print_system_info() -> None:
    """打印系统信息"""
    from src.core.config import CONFIG
    
    info = get_system_info()
    
    logger.info("="*50)
    logger.info(" 图像超分辨率API系统信息")
    logger.info("="*50)
    logger.info(f"操作系统:     {info['os']}")
    logger.info(f"Python版本:   {info['python_version']}")
    logger.info(f"PyTorch版本:  {info['torch_version']}")
    logger.info(f"CPU信息:      {info['cpu']}")
    logger.info(f"CPU核心:      {info['cpu_cores']} 物理核心, {info['cpu_threads']} 逻辑核心")
    logger.info(f"系统内存:     {info['memory']}")
    
    if info["gpu_available"] == "No":
        logger.info("GPU支持:      不可用 (使用CPU模式)")
    elif info["gpu_available"] == "CUDA":
        logger.info(f"GPU支持:      可用 (CUDA)")
        logger.info(f"GPU数量:      {info['gpu_count']}")
        for i, gpu in enumerate(info["gpu_info"]):
            logger.info(f"  GPU {i}: {gpu['name']} ({gpu['memory']})")
    elif info["gpu_available"] == "MPS (Apple Metal)":
        logger.info("GPU支持:      可用 (Apple Metal/MPS)")
    
    logger.info("="*50)
    logger.info(f"模型目录:     {os.path.abspath(CONFIG['model_dir'])}")
    logger.info(f"临时文件目录: {CONFIG['temp_dir']}")
    logger.info(f"最大并发任务: {CONFIG['max_concurrent_tasks']}")
    logger.info("="*50)

def get_device():
    """自动选择最佳计算设备"""
    from src.core.config import CONFIG
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 检查Mac MPS (Metal Performance Shaders)是否可用
    if CONFIG["use_mps_if_available"] and platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # 默认使用CPU
    return torch.device("cpu")

def get_memory_usage() -> Dict[str, Any]:
    """获取当前内存使用情况"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percent": memory.percent
    }

# 导入os
import os
