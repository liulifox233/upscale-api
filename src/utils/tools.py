"""
工具模块：提供各种辅助功能，如URL验证、图像下载等
"""
import io
import os
import shutil
import logging
import requests
from urllib.parse import urlparse
from fastapi import HTTPException
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger("upscale-api")

def is_valid_url(url: str) -> bool:
    """验证URL是否有效
    
    Args:
        url: 要验证的URL
        
    Returns:
        URL是否有效
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image(url: str, allowed_mime_types: set, headers: Optional[Dict[str, Any]] = None) -> Image.Image:
    """从URL下载图片
    
    Args:
        url: 图片URL
        allowed_mime_types: 允许的MIME类型
        headers: 要转发的请求头
        
    Returns:
        下载的图片对象
        
    Raises:
        HTTPException: 下载失败或图片类型不支持
    """
    try:
        # 过滤掉一些不需要转发的头信息
        if headers:
            # 移除host、content-length等不应该转发的头
            filtered_headers = {k: v for k, v in headers.items() 
                               if k.lower() not in ['host', 'content-length', 'connection']}
            logger.info(f"使用以下头信息请求图像: {filtered_headers}")
            response = requests.get(url, stream=True, headers=filtered_headers)
        else:
            response = requests.get(url, stream=True)
            
        response.raise_for_status()
        
        # 验证内容类型
        content_type = response.headers.get('content-type', '')
        if not any(mime in content_type for mime in allowed_mime_types):
            raise ValueError(f"Unsupported image type: {content_type}")
            
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.error(f"下载图片失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def cleanup_temp_files(temp_dir: str) -> None:
    """清理临时文件
    
    Args:
        temp_dir: 临时文件目录
    """
    try:
        logger.info(f"清理临时文件目录: {temp_dir}")
        # 清空目录但不删除目录本身
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"清理文件失败: {file_path}, 错误: {e}")
        logger.info("临时文件清理完成")
    except Exception as e:
        logger.error(f"清理临时文件时出错: {str(e)}", exc_info=True)
