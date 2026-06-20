"""
Logging Utilities
日志工具
"""

import os
import logging


def get_logger(log_path):
    """
    创建logger对象，同时输出到文件和控制台
    
    Args:
        log_path: 日志文件路径
    
    Returns:
        logger对象
    """
    parent_path = os.path.dirname(log_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    
    return logger
