"""
API路由模块：定义所有的API端点
"""
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image
import io
from typing import Optional

from src.core.config import CONFIG
from src.utils.system_info import get_device, get_memory_usage
from src.models.model_manager import get_available_models
from src.utils.tools import is_valid_url, download_image
from src.core.image_processor import process_image, get_processing_semaphore
from src.utils.image_cache import get_image_cache

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
    request: Request,
    file: UploadFile = File(None),
    url: str = Query(None, description="Image URL to download and process"),
    model: str = Query(CONFIG["default_model"], description="Model to use for upscaling"),
    format: str = Query(CONFIG["default_format"], description="Output image format"),
    quality: int = Query(90, ge=1, le=100, description="Output quality (1-100, for JPG/WEBP)"),
    half_precision: bool = Query(None, description="是否启用FP16半精度量化（覆盖全局配置）")
):
    """
    图片超分辨率处理API (GET方法)
    
    参数:
    - file: 上传的图片文件 (可选)
    - url: 要处理的图片URL (可选)
    - model: 使用的模型名称 (默认: RealESRGAN)
    - format: 输出格式 [png, jpg, webp] (默认: png)
    - quality: 输出质量 (1-100, 仅对jpg/webp有效)
    
    注意: 必须提供file或url中的一个
    """
    return await _process_upscale(request, file, url, model, format, quality, half_precision)

@router.post("/upscale/", response_class=FileResponse)
async def upscale_image_post(
    request: Request,
    model: str = Query(CONFIG["default_model"], description="Model to use for upscaling"),
    format: str = Query(CONFIG["default_format"], description="Output image format"),
    quality: int = Query(90, ge=1, le=100, description="Output quality (1-100, for JPG/WEBP)"),
    half_precision: bool = Query(None, description="是否启用FP16半精度量化（覆盖全局配置）"),
    url: str = Query(None, description="Image URL to download and process")
):
    """
    图片超分辨率处理API (POST方法)
    
    参数:
    - 请求体: 原始图片数据 (Content-Type: application/octet-stream 或其他图片类型)
    - model: 使用的模型名称 (默认: RealESRGAN)
    - format: 输出格式 [png, jpg, webp] (默认: png)
    - quality: 输出质量 (1-100, 仅对jpg/webp有效)
    - url: 要处理的图片URL (可选)
    
    注意: 必须提供请求体中的图片数据或url参数中的一个
    """
    logger.info("处理POST请求的图片超分辨率处理")
    file = None
    body_data = await request.body()
    
    if body_data and not url:
        # 获取内容类型，默认为application/octet-stream
        content_type = request.headers.get("content-type", "application/octet-stream")
        
        if content_type == "application/octet-stream":
            # 根据图片头部字节尝试判断图片类型
            if body_data.startswith(b'\xff\xd8\xff'):  # JPEG
                content_type = "image/jpeg"
            elif body_data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                content_type = "image/png"
            elif body_data.startswith(b'RIFF') and b'WEBP' in body_data[0:12]:  # WEBP
                content_type = "image/webp"
            else:
                content_type = "image/jpeg"
                
            logger.info(f"收到application/octet-stream数据，推断内容类型为: {content_type}")
        
        class TempUploadFile:
            def __init__(self, content, content_type):
                self.content_type = content_type
                self._content = content
                self.filename = "uploaded_image"
            
            async def read(self):
                return self._content
        
        file = TempUploadFile(body_data, content_type)
    return await _process_upscale(request, file, url, model, format, quality, half_precision)

async def _process_upscale(
    request: Request,
    file,
    url: str,
    model: str,
    format: str,
    quality: int,
    half_precision: Optional[bool] = None
):
    """
    处理上传的图片或URL图片的共享逻辑
    """
    try:
        # 获取请求头
        headers = dict(request.headers)
        logger.info(f"接收到请求，headers: {headers}")
        
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
        
        # 如果是URL请求，检查缓存
        if url:
            if not is_valid_url(url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL provided"
                )
            
            # 检查缓存
            image_cache = get_image_cache()
            cached_path = image_cache.get(url, model, format, quality)
            
            if cached_path:
                logger.info(f"使用缓存的处理结果: {cached_path}")
                return FileResponse(
                    cached_path,
                    media_type=CONFIG["supported_output_formats"][format],
                    filename=os.path.basename(cached_path)
                )
            
            logger.info(f"从URL下载图像: {url}")
            image = download_image(url, CONFIG["allowed_mime_types"], headers)
        
        # 处理文件上传
        else:
            # 验证文件类型
            content_type = file.content_type
            
            # 特殊处理application/octet-stream
            if content_type == "application/octet-stream":
                logger.info("收到application/octet-stream类型的数据，尝试作为图像处理")
            elif content_type not in CONFIG["allowed_mime_types"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed types: {', '.join(CONFIG['allowed_mime_types'])}"
                )
            
            logger.info(f"处理上传的文件: {getattr(file, 'filename', 'unknown')}")
            # 读取上传的图片
            contents = await file.read()
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as e:
                logger.error(f"无法打开图像: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image data: {str(e)}"
                )
        
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
        
        # 如果是URL请求，将结果添加到缓存
        if url:
            image_cache = get_image_cache()
            cached_path = image_cache.put(url, model, format, quality, output_path)
            output_path = cached_path  # 使用缓存的路径
        
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
