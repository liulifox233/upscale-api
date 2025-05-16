"""
模型管理模块：负责模型的加载和缓存
"""
import os
import glob
import logging
import torch
from typing import List, Optional
from spandrel import ModelLoader

from src.utils.system_info import get_device
from src.utils.memory_manager import get_memory_manager

logger = logging.getLogger("upscale-api")

def get_available_models(model_dir: str) -> List[str]:
    """获取可用的模型列表
    
    Args:
        model_dir: 存储模型的目录路径
    
    Returns:
        模型名称列表
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return []
    
    # 查找所有.pth文件
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    return [os.path.splitext(os.path.basename(m))[0] for m in model_files]

def load_model(model_name: str, model_dir: str, use_half_precision: bool) -> Optional[torch.nn.Module]:
    """加载指定模型
    
    Args:
        model_name: 模型名称
        model_dir: 存储模型的目录路径
        use_half_precision: 是否使用半精度量化
        
    Returns:
        加载的模型实例
    """
    # 获取内存管理器
    memory_manager = get_memory_manager()
    
    # 先从缓存中查找
    model = memory_manager.get_cached_model(model_name)
    if model:
        logger.info(f"从缓存加载模型 {model_name}")
        return model
    
    try:
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        # 从本地文件加载模型
        logger.info(f"从本地文件加载模型 {model_name}")
        model_descriptor = ModelLoader().load_from_file(model_path)
        
        model = model_descriptor.model
        
        # 将模型移动到适当的设备
        device = get_device()
        
        # 根据配置决定是否使用半精度量化
        if use_half_precision:
            logger.info(f"使用FP16半精度量化模式")
            model = model.half().to(device)
        else:
            logger.info(f"使用FP32全精度模式")
            model = model.to(device)
            
        # 缓存模型
        memory_manager.cache_model(model_name, model)
        return model
        
    except Exception as e:
        logger.error(f"加载模型 {model_name} 失败: {str(e)}")
        return None
