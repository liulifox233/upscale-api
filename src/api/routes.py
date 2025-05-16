"""
API路由模块：定义所有的API端点
"""
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
import io
from typing import Optional

from src.core.config import CONFIG
from src.utils.system_info import get_device, get_memory_usage
from src.models.model_manager import get_available_models
from src.utils.tools import is_valid_url, download_image
from src.core.image_processor import process_image, get_processing_semaphore

router = APIRouter()
logger = logging.getLogger("upscale-api")

@router.get("/models")
async def list_available_models():
    """获取可用模型列表"""
    available_models = get_available_models(CONFIG["model_dir"])
    pretrained_models = ["RealESRGAN", "ESRGAN", "SwinIR"]  # 示例预训练模型
    
    return {
        "local_models": available_models,
        "pretrained_models": pretrained_models,
        "default_model": CONFIG["default_model"],
        "current_device": str(get_device())
    }

@router.get("/upscale/", response_class=FileResponse)
async def upscale_image(
    file: UploadFile = File(None),
    url: str = Query(None, description="Image URL to download and process"),
    model: str = Query(CONFIG["default_model"], description="Model to use for upscaling"),
    format: str = Query(CONFIG["default_format"], description="Output image format"),
    quality: int = Query(90, ge=1, le=100, description="Output quality (1-100, for JPG/WEBP)"),
    half_precision: bool = Query(None, description="是否启用FP16半精度量化（覆盖全局配置）")
):
    """
    图片超分辨率处理API
    
    参数:
    - file: 上传的图片文件 (可选)
    - url: 要处理的图片URL (可选)
    - model: 使用的模型名称 (默认: RealESRGAN)
    - format: 输出格式 [png, jpg, webp] (默认: png)
    - quality: 输出质量 (1-100, 仅对jpg/webp有效)
    
    注意: 必须提供file或url中的一个
    """
    try:
        # 验证输入
        if not file and not url:
            raise HTTPException(
                status_code=400,
                detail="Either file or url must be provided"
            )
        
        if file and url:
            raise HTTPException(
                status_code=400,
                detail="Provide either file or url, not both"
            )
        
        # 验证输出格式
        format = format.lower()
        if format not in CONFIG["supported_output_formats"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format. Supported: {', '.join(CONFIG['supported_output_formats'].keys())}"
            )
        
        # 处理文件上传
        if file:
            # 验证文件类型
            if file.content_type not in CONFIG["allowed_mime_types"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed types: {', '.join(CONFIG['allowed_mime_types'])}"
                )
            
            logger.info(f"处理上传的文件: {file.filename}")
            # 读取上传的图片
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 处理URL
        else:
            if not is_valid_url(url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL provided"
                )
            
            logger.info(f"从URL下载图像: {url}")
            image = download_image(url, CONFIG["allowed_mime_types"])
        
        # 临时覆盖配置，如果指定了半精度参数
        config_override = None
        if half_precision is not None:
            config_override = CONFIG.copy()
            config_override["use_half_precision"] = half_precision
        
        # 处理图片
        output_path = await process_image(
            image=image,
            model_name=model,
            output_format=format,
            quality=quality,
            config=config_override
        )

        # 关闭PIL图像对象，释放内存
        image.close()
        
        # 返回处理后的图片
        return FileResponse(
            output_path,
            media_type=CONFIG["supported_output_formats"][format],
            filename=os.path.basename(output_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """获取系统状态"""
    memory = get_memory_usage()
    
    # 获取当前可用的处理槽
    semaphore = get_processing_semaphore()
    available_slots = semaphore._value
    
    return {
        "status": "running",
        "memory_usage": f"{memory['percent']}%",
        "available_processing_slots": available_slots,
        "queued_tasks": CONFIG["max_concurrent_tasks"] - available_slots,
        "device": str(get_device())
    }
