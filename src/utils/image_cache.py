"""
图像缓存模块：提供图像处理结果的缓存功能
"""
import os
import logging
import hashlib
import shutil
from PIL import Image
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("upscale-api")

class ImageCache:
    """图像缓存管理器
    
    根据URL或其它标识符缓存处理后的图像，存储在本地文件系统
    """
    
    def __init__(self, cache_dir: str):
        """初始化图像缓存
        
        Args:
            cache_dir: 缓存目录路径（与临时文件共享同一目录）
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, str] = {}  # 映射缓存键到文件路径
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"图像缓存初始化，使用目录: {cache_dir}")
    
    def _generate_key(self, url: str, model_name: str, output_format: str, quality: int) -> str:
        """生成缓存键
        
        Args:
            url: 图像URL
            model_name: 使用的模型名称
            output_format: 输出格式
            quality: 输出质量
            
        Returns:
            缓存键
        """
        # 组合所有参数创建键
        key_str = f"{url}:{model_name}:{output_format}:{quality}"
        # 使用MD5生成唯一标识
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, url: str, model_name: str, output_format: str, quality: int) -> Optional[str]:
        """获取缓存的图像路径
        
        Args:
            url: 图像URL
            model_name: 使用的模型名称
            output_format: 输出格式
            quality: 输出质量
            
        Returns:
            缓存的图像路径，如果不存在则返回None
        """
        key = self._generate_key(url, model_name, output_format, quality)
        
        if key in self.cache:
            cached_path = self.cache[key]
            if os.path.exists(cached_path):
                logger.info(f"缓存命中：{url}")
                return cached_path
            else:
                # 文件不存在，从缓存中移除
                logger.warning(f"缓存文件不存在，删除缓存记录: {cached_path}")
                del self.cache[key]
        
        logger.info(f"缓存未命中：{url}")
        return None
    
    def put(self, url: str, model_name: str, output_format: str, quality: int, 
            image_path: str) -> str:
        """添加图像到缓存
        
        Args:
            url: 图像URL
            model_name: 使用的模型名称
            output_format: 输出格式
            quality: 输出质量
            image_path: 图像文件路径
            
        Returns:
            缓存的图像路径
        """
        key = self._generate_key(url, model_name, output_format, quality)
        
        # 创建缓存文件名
        cache_filename = f"cache_{key}.{output_format}"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # 复制图像到缓存目录
        try:
            shutil.copy2(image_path, cache_path)
            self.cache[key] = cache_path
            logger.info(f"图像已缓存: {url} -> {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"缓存图像失败: {str(e)}")
            return image_path  # 如果缓存失败，返回原始路径
    
    def clear(self) -> None:
        """清空缓存"""
        logger.info("清空图像缓存...")
        
        # 删除所有缓存文件
        for key, path in self.cache.items():
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"已删除缓存文件: {path}")
            except Exception as e:
                logger.error(f"删除缓存文件失败: {path}, 错误: {str(e)}")
        
        # 清空缓存字典
        self.cache.clear()
        logger.info("图像缓存已清空")


# 全局缓存实例，将在应用启动时初始化
_image_cache: Optional[ImageCache] = None

def init_image_cache(cache_dir: str) -> None:
    """初始化全局图像缓存
    
    Args:
        cache_dir: 缓存目录路径
    """
    global _image_cache
    _image_cache = ImageCache(cache_dir)

def get_image_cache() -> ImageCache:
    """获取全局图像缓存实例
    
    Returns:
        图像缓存实例
    
    Raises:
        RuntimeError: 如果缓存尚未初始化
    """
    if _image_cache is None:
        raise RuntimeError("图像缓存尚未初始化")
    return _image_cache