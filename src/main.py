from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from spandrel import ModelLoader
from PIL import Image
import numpy as np
import io
import os
import uuid
import tempfile
import requests
from urllib.parse import urlparse
import torch
from typing import List
import glob
import platform
import psutil
import sys
import gc
import logging
import asyncio
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("upscale-api")

app = FastAPI(title="Advanced Image Upscaling API")

# 配置
CONFIG = {
    "model_dir": "weights",  # 模型权重目录
    "temp_dir": tempfile.gettempdir(),  # 临时文件目录
    "allowed_mime_types": {  # 允许的输入图片类型
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff"
    },
    "supported_output_formats": {  # 支持的输出格式
        "png": "image/png",
        "jpg": "image/jpeg",
        "webp": "image/webp"
    },
    "default_model": "RealESRGAN",  # 默认模型
    "default_format": "webp",  # 默认输出格式
    "use_mps_if_available": True,  # 是否启用Mac GPU支持
    "use_half_precision": True,  # 是否启用FP16半精度量化
    "max_concurrent_tasks": 5,  # 最大并发处理数量
    "memory_threshold": 80,  # 内存使用率阈值，超过此值将触发内存回收（百分比）
    "cleanup_interval": 5,  # 清理间隔（处理N次图像后执行一次清理）
}

# 全局模型缓存
MODEL_CACHE = {}

# 并发控制信号量
processing_semaphore = asyncio.Semaphore(CONFIG["max_concurrent_tasks"])

# 处理计数器，用于定期清理
processing_counter = 0

def get_system_info():
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

def print_system_info():
    """打印系统信息"""
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 检查Mac MPS (Metal Performance Shaders)是否可用
    if CONFIG["use_mps_if_available"] and platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # 默认使用CPU
    return torch.device("cpu")

def get_memory_usage():
    """获取当前内存使用情况"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percent": memory.percent
    }

def cleanup_memory():
    """清理内存"""
    global processing_counter
    
    # 获取内存使用情况
    memory = get_memory_usage()
    
    # 如果达到计数器阈值或内存使用率超过阈值，执行清理
    if (processing_counter >= CONFIG["cleanup_interval"] or 
            memory["percent"] > CONFIG["memory_threshold"]):
        
        logger.info(f"执行内存清理 (内存使用率: {memory['percent']}%)")
        
        # 强制执行Python垃圾收集
        gc.collect()
        
        # 如果有CUDA设备，清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        processing_counter = 0
        
        # 清理后的内存状态
        new_memory = get_memory_usage()
        logger.info(f"内存清理后: {new_memory['percent']}%")
    else:
        processing_counter += 1

def get_available_models() -> List[str]:
    """获取可用的模型列表"""
    if not os.path.exists(CONFIG["model_dir"]):
        os.makedirs(CONFIG["model_dir"])
        return []
    
    # 查找所有.pth文件
    model_files = glob.glob(os.path.join(CONFIG["model_dir"], "*.pth"))
    return [os.path.splitext(os.path.basename(m))[0] for m in model_files]

def load_model(model_name: str):
    """加载指定模型"""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    model_path = os.path.join(CONFIG["model_dir"], f"{model_name}.pth")
    # 从本地文件加载模型
    logger.info(f"从本地文件加载模型 {model_name}")
    model = ModelLoader().load_from_file(model_path)
    # 将模型移动到适当的设备
    device = get_device()
    
    # 根据配置决定是否使用半精度量化
    if CONFIG["use_half_precision"]:
        logger.info(f"使用FP16半精度量化模式")
        model = model.half().to(device)
    else:
        logger.info(f"使用FP32全精度模式")
        model = model.to(device)
        
    MODEL_CACHE[model_name] = model
    return model

def is_valid_url(url: str) -> bool:
    """验证URL是否有效"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image(url: str) -> Image.Image:
    """从URL下载图片"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 验证内容类型
        content_type = response.headers.get('content-type', '')
        if not any(mime in content_type for mime in CONFIG["allowed_mime_types"]):
            raise ValueError("Unsupported image type")
            
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@asynccontextmanager
async def processing_limiter():
    """限制并发处理任务的上下文管理器"""
    await processing_semaphore.acquire()
    try:
        yield
    finally:
        processing_semaphore.release()

async def process_image(
    image: Image.Image,
    model_name: str,
    output_format: str,
    quality: int = 90
) -> str:
    """
    处理图片并返回保存路径
    
    参数:
    - image: 输入图片
    - model_name: 使用的模型名称
    - output_format: 输出格式 (png/jpg/webp)
    - quality: 输出质量 (1-100)
    """
    try:
        async with processing_limiter():
            logger.info(f"开始处理图像，使用模型: {model_name}")
            
            model = load_model(model_name)
            device = get_device()
        
            # 将图片转换为numpy数组并调整维度
            img_array = np.array(image)
            
            # 转换维度顺序为CHW (Channels, Height, Width)
            img_array = np.transpose(img_array, (2, 0, 1))  # 从HWC转为CHW
            
            # 添加batch维度 (变为NCHW)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 转换为torch张量并归一化到[0,1]范围
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            
            # 根据配置决定是否使用半精度量化
            if CONFIG["use_half_precision"]:
                img_tensor = img_tensor.half().to(device)
            else:
                img_tensor = img_tensor.to(device)
            
            # 使用模型进行超分辨率处理
            with torch.no_grad():
                logger.info(f"图像尺寸: {image.size}，开始超分处理")
                output = model(img_tensor) # type: ignore
            
            # 将输出转换回PIL图像
            output = output.clamp(0, 1) * 255.0  # 反归一化到[0,255]
            output = output.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()  # 转为HWC格式
            output_img = Image.fromarray(output)
            
            # 生成唯一的文件名
            unique_filename = f"upscaled_{uuid.uuid4().hex}.{output_format}"
            output_path = os.path.join(CONFIG["temp_dir"], unique_filename)
            
            # 保存处理后的图片
            save_kwargs = {}
            if output_format in ["jpg", "webp"]:
                save_kwargs["quality"] = quality
            
            output_img.save(output_path, **save_kwargs)
            
            # 执行内存清理
            cleanup_memory()
            
            logger.info(f"处理完成，输出文件: {output_path}")
            return output_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    
@app.get("/models")
async def list_available_models():
    """获取可用模型列表"""
    available_models = get_available_models()
    pretrained_models = ["RealESRGAN", "ESRGAN", "SwinIR"]  # 示例预训练模型
    
    return {
        "local_models": available_models,
        "pretrained_models": pretrained_models,
        "default_model": CONFIG["default_model"],
        "current_device": str(get_device())
    }

@app.get("/upscale/", response_class=FileResponse)
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
            image = download_image(url)
        
        # 处理图片
        output_path = await process_image(
            image=image,
            model_name=model,
            output_format=format,
            quality=quality
        )
        
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

@app.get("/status")
async def get_status():
    """获取系统状态"""
    memory = get_memory_usage()
    available_slots = processing_semaphore._value
    
    return {
        "status": "running",
        "memory_usage": f"{memory['percent']}%",
        "available_processing_slots": available_slots,
        "queued_tasks": CONFIG["max_concurrent_tasks"] - available_slots,
        "device": str(get_device())
    }

if __name__ == "__main__":
    # 启动时打印系统信息
    print_system_info()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)