"""
内存管理模块：处理内存清理和垃圾回收
"""
import gc
import logging
import torch
import weakref
from typing import Dict, Any, Optional

from src.utils.system_info import get_memory_usage

logger = logging.getLogger("upscale-api")

# 使用类来管理模型缓存和处理计数
class MemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_counter = 0
        # 使用 weakref.WeakValueDictionary 替代普通字典存储模型
        # 这样当模型不再被使用时，它们可以被垃圾回收
        self._model_cache = weakref.WeakValueDictionary()
        logger.info("内存管理器初始化完成")
        
    def get_cached_model(self, model_name: str):
        """从缓存中获取模型"""
        return self._model_cache.get(model_name)
        
    def cache_model(self, model_name: str, model_instance):
        """缓存模型，并管理缓存大小"""
        # 检查缓存大小
        if len(self._model_cache) >= self.config["max_model_cache_size"]:
            # 如果已达到最大缓存大小，移除最早加入的模型
            try:
                # 获取第一个键（最早的模型）
                oldest_key = next(iter(self._model_cache.keys()))
                # 从缓存中弹出
                logger.info(f"缓存已满，移除最早的模型: {oldest_key}")
                del self._model_cache[oldest_key]
            except (StopIteration, KeyError):
                # 如果没有模型或者模型已经被回收，则跳过
                pass
                
        # 缓存新模型
        self._model_cache[model_name] = model_instance
        logger.info(f"模型 {model_name} 已添加到缓存，当前缓存大小: {len(self._model_cache)}")
        
    def cleanup_memory(self, force: bool = False) -> None:
        """清理内存
        
        Args:
            force: 是否强制清理，忽略计数器和内存阈值
        """
        # 获取内存使用情况
        memory = get_memory_usage()
        
        # 如果达到计数器阈值、内存使用率超过阈值，或强制清理，则执行清理
        if (force or 
            self.processing_counter >= self.config["cleanup_interval"] or 
            memory["percent"] > self.config["memory_threshold"]):
            
            logger.info(f"执行内存清理 (内存使用率: {memory['percent']}%)")
            
            # 强制执行Python垃圾回收
            gc.collect()
            
            # 如果有CUDA设备，清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.processing_counter = 0
            
            # 清理后的内存状态
            new_memory = get_memory_usage()
            logger.info(f"内存清理后: {new_memory['percent']}%，当前缓存模型数: {len(self._model_cache)}")
        else:
            self.processing_counter += 1
            
    def cleanup_model_cache(self) -> None:
        """完全清理模型缓存"""
        self._model_cache.clear()
        # 强制执行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型缓存已完全清理")

# 全局内存管理器实例，将在应用启动时初始化
memory_manager: Optional[MemoryManager] = None

def init_memory_manager(config: Dict[str, Any]) -> None:
    """初始化全局内存管理器"""
    global memory_manager
    memory_manager = MemoryManager(config)

def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器实例"""
    if memory_manager is None:
        raise RuntimeError("内存管理器尚未初始化")
    return memory_manager
