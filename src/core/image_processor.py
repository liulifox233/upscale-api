"""
图像处理模块：处理图像超分辨率
"""
import os
import logging
import tempfile
import asyncio
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional, Union
from fastapi import HTTPException

from src.utils.system_info import get_device
from src.utils.memory_manager import get_memory_manager
from src.models.model_manager import load_model

logger = logging.getLogger("upscale-api")

# 并发控制信号量，将在应用启动时初始化
processing_semaphore: Optional[asyncio.Semaphore] = None

def init_processing_semaphore(max_concurrent_tasks: int) -> None:
    """初始化全局并发控制信号量"""
    global processing_semaphore
    processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)

def get_processing_semaphore() -> asyncio.Semaphore:
    """获取全局并发控制信号量"""
    if processing_semaphore is None:
        raise RuntimeError("并发控制信号量尚未初始化")
    return processing_semaphore

async def process_image(
    image: Image.Image,
    model_name: str,
    output_format: str,
    quality: int = 90,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    处理图片并返回保存路径
    
    参数:
    - image: 输入图片
    - model_name: 使用的模型名称
    - output_format: 输出格式 (png/jpg/webp)
    - quality: 输出质量 (1-100)
    - config: 配置项
    """
    if config is None:
        from src.core.config import CONFIG
        config = CONFIG
        
    output_path = None
    
    # 获取内存管理器
    memory_manager = get_memory_manager()
    # 获取并发控制信号量
    semaphore = get_processing_semaphore()
    
    try:
        # 使用信号量限制并发处理
        async with semaphore:
            logger.info(f"开始处理图像，使用模型: {model_name}")
            
            # 加载模型
            model = load_model(
                model_name=model_name, 
                model_dir=config["model_dir"],
                use_half_precision=config["use_half_precision"]
            )
            
            # 若模型加载失败
            if model is None:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {model_name}")
                
            device = get_device()
        
            try:
                # 将图片转换为numpy数组并调整维度
                img_array = np.array(image)
                
                # 转换维度顺序为CHW (Channels, Height, Width)
                img_array = np.transpose(img_array, (2, 0, 1))  # 从HWC转为CHW
                
                # 添加batch维度 (变为NCHW)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 转换为torch张量并归一化到[0,1]范围
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                
                # 根据配置决定是否使用半精度量化
                if config["use_half_precision"]:
                    img_tensor = img_tensor.half().to(device)
                else:
                    img_tensor = img_tensor.to(device)
                
                logger.info(f"图像尺寸: {image.size}，开始超分处理")
                
                # 使用模型进行超分辨率处理
                with torch.no_grad():
                    output = model(img_tensor)
                
                # 将输出转换回PIL图像
                output = output.clamp(0, 1) * 255.0  # 反归一化到[0,255]
                output = output.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()  # 转为HWC格式
                output_img = Image.fromarray(output)
                
                # 生成临时文件路径
                with tempfile.NamedTemporaryFile(
                    suffix=f".{output_format}", 
                    prefix="upscaled_",
                    dir=config["temp_dir"],
                    delete=False
                ) as temp_file:
                    output_path = temp_file.name
                
                # 保存处理后的图片
                save_kwargs = {}
                if output_format in ["jpg", "webp"]:
                    save_kwargs["quality"] = quality
                
                output_img.save(output_path, **save_kwargs)
                
                # 关闭PIL图像对象，释放内存
                output_img.close()
                
                # 显式释放大型变量，帮助垃圾回收
                del img_array, img_tensor, output
                
            except Exception as e:
                logger.error(f"图像处理失败: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
            finally:
                # 执行内存清理
                memory_manager.cleanup_memory()
            
            logger.info(f"处理完成，输出文件: {output_path}")
            return output_path

    except HTTPException:
        # 如果处理过程中出错且已创建临时文件，则删除它
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"由于处理错误，已删除临时文件: {output_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {str(e)}")
        raise
