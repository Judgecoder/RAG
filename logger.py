import os
import logging
from logging.handlers import RotatingFileHandler

# 确保日志文件夹存在
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 创建日志记录器
logger = logging.getLogger('RAG')
logger.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器（支持日志轮转）
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app.log'),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5  # 最多保留5个备份
)
file_handler.setLevel(logging.DEBUG)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器到记录器
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# 导出logger实例
__all__ = ['logger']