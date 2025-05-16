"""
应用初始化模块：负责应用程序的初始化和配置
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.core.config import CONFIG, TEMP_OUTPUT_DIR
from src.utils.system_info import print_system_info
from src.utils.memory_manager import init_memory_manager, get_memory_manager
from src.core.image_processor import init_processing_semaphore
from src.utils.tools import cleanup_temp_files
from src.api.routes import router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("upscale-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理，处理启动和关闭事件"""
    # 启动时执行的操作
    logger.info("初始化应用程序...")
    
    # 确保临时目录存在
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    
    # 打印系统信息
    print_system_info()
    
    # 初始化内存管理器
    init_memory_manager(CONFIG)
    
    # 初始化并发控制信号量
    init_processing_semaphore(CONFIG["max_concurrent_tasks"])
    
    logger.info(f"创建临时输出目录: {TEMP_OUTPUT_DIR}")
    logger.info("应用程序初始化完成")
    
    yield  # 应用程序运行期间
    
    # 应用程序关闭时执行的清理操作
    logger.info("应用程序关闭，执行清理操作...")
    
    # 清理模型缓存
    memory_manager = get_memory_manager()
    memory_manager.cleanup_model_cache()
    
    # 清理临时文件
    cleanup_temp_files(TEMP_OUTPUT_DIR)
    
    logger.info("清理操作完成，应用程序已关闭")

def create_application() -> FastAPI:
    """创建FastAPI应用程序实例"""
    app = FastAPI(
        title="Advanced Image Upscaling API",
        description="用于图像超分辨率处理的API服务",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 注册路由
    app.include_router(router)
    
    return app

# 创建应用实例
app = create_application()
