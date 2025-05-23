"""
配置模块：包含应用程序的所有配置项
"""
import os
import tempfile

# 创建临时目录用于存储处理后的图片和缓存
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "upscale_api_outputs")
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

# 应用程序配置
CONFIG = {
    "model_dir": "weights",  # 模型权重目录
    "temp_dir": TEMP_OUTPUT_DIR,  # 临时文件目录（同时用作缓存目录）
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
    "max_concurrent_tasks": 10,  # 最大并发处理数量
    "memory_threshold": 0,  # 内存使用率阈值，超过此值将触发内存回收（百分比）
    "cleanup_interval": 10,  # 清理间隔（处理N次图像后执行一次清理）
    "max_model_cache_size": 3,  # 模型缓存大小限制,
    "auto_color_model": "2x_IllustrationJaNai_V1_ESRGAN_120k",  # 自动模式下用于彩色图像的模型
    "auto_gray_model": "2x_MangaJaNai_1500p_V1_ESRGAN_90k",  # 自动模式下用于灰度图像的模型
    "high_res_threshold": 1500,  # 高分辨率阈值，高于此分辨率的彩色图像将不进行超分辨率处理
}
