import os
import logging

def setup_logging(log_file_name):
    # 创建日志目录
    log_dir = './Logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件的完整路径
    log_file_path = os.path.join(log_dir, log_file_name)

    # 创建 logger
    logger = logging.getLogger(log_file_name)  # 使用文件名作为 logger 的名称
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 检查是否已经有 handler 配置
    if logger.hasHandlers():
        logger.handlers.clear()  # 清除现有的 handlers

    # 创建一个 file handler 写日志到文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # 创建一个 stream handler 输出日志到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_format = logging.Formatter('%(message)s')  # 控制台输出不需要太详细
    stream_handler.setFormatter(stream_format)

    # 添加 handlers 到 logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger