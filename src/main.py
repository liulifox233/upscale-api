import uvicorn
import sys
import os

# 添加项目根目录到Python路径，以便能够导入src模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入应用实例
from src.core.application import app

if __name__ == "__main__":
    # 启动 FastAPI 应用
    uvicorn.run(app, host="0.0.0.0", port=8000)
