"""
日誌管理模組
提供統一的日誌記錄功能，標準化錯誤處理和記錄方式
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import LogConfig, BASE_DIR


class AppLogger:
    """應用程式日誌管理器"""
    
    _loggers = {}
    _log_file: Optional[Path] = None
    
    @classmethod
    def setup_log_file(cls, log_dir: Optional[Path] = None) -> Path:
        """
        設定日誌檔案路徑
        
        Args:
            log_dir: 日誌目錄，預設為 BASE_DIR/logs
            
        Returns:
            Path: 日誌檔案的完整路徑
        """
        if log_dir is None:
            log_dir = BASE_DIR / "logs"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 產生帶時間戳記的日誌檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"app_{timestamp}.log"
        
        cls._log_file = log_file
        return log_file
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        取得或建立指定名稱的 Logger
        
        Args:
            name: Logger 名稱（通常使用 __name__）
            level: 日誌等級（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            
        Returns:
            logging.Logger: 設定好的 Logger 實例
        """
        # 如果已經存在，直接回傳
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 建立新的 Logger
        logger = logging.getLogger(name)
        
        # 設定日誌等級
        log_level = level or LogConfig.LOG_LEVEL
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重複添加 Handler
        if not logger.handlers:
            # 建立格式化器
            formatter = logging.Formatter(
                LogConfig.LOG_FORMAT,
                datefmt=LogConfig.LOG_DATE_FORMAT
            )
            
            # 控制台 Handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 檔案 Handler（如果有設定日誌檔案）
            if cls._log_file is not None:
                file_handler = logging.FileHandler(cls._log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        # 避免日誌傳播到父 Logger
        logger.propagate = False
        
        # 快取 Logger
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def log_exception(cls, logger: logging.Logger, exception: Exception, context: str = ""):
        """
        記錄例外資訊（包含堆疊追蹤）
        
        Args:
            logger: Logger 實例
            exception: 例外物件
            context: 額外的上下文資訊
        """
        error_msg = f"{context}: {str(exception)}" if context else str(exception)
        logger.error(error_msg, exc_info=True)
    
    @classmethod
    def log_function_call(cls, logger: logging.Logger, func_name: str, **kwargs):
        """
        記錄函數呼叫資訊
        
        Args:
            logger: Logger 實例
            func_name: 函數名稱
            **kwargs: 函數參數
        """
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"呼叫函數: {func_name}({params})")
    
    @classmethod
    def log_performance(cls, logger: logging.Logger, operation: str, duration: float):
        """
        記錄效能資訊
        
        Args:
            logger: Logger 實例
            operation: 操作名稱
            duration: 執行時間（秒）
        """
        logger.info(f"效能: {operation} 耗時 {duration:.2f} 秒")


# 初始化日誌檔案
AppLogger.setup_log_file()


# ============================================================================
# 便利函數
# ============================================================================

def get_logger(name: str = __name__) -> logging.Logger:
    """
    快速取得 Logger 的便利函數
    
    Args:
        name: Logger 名稱
        
    Returns:
        logging.Logger: Logger 實例
    """
    return AppLogger.get_logger(name)


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    記錄錯誤的便利函數
    
    Args:
        logger: Logger 實例
        error: 錯誤物件
        context: 上下文資訊
    """
    AppLogger.log_exception(logger, error, context)


# ============================================================================
# Streamlit 整合
# ============================================================================

def streamlit_error_handler(func):
    """
    裝飾器：捕捉函數錯誤並顯示在 Streamlit 介面上
    
    使用方式:
        @streamlit_error_handler
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(logger, e, f"執行 {func.__name__} 時發生錯誤")
            
            # 如果在 Streamlit 環境中，顯示錯誤訊息
            try:
                import streamlit as st
                st.error(f"❌ 發生錯誤: {str(e)}")
                with st.expander("詳細錯誤資訊"):
                    st.code(f"{type(e).__name__}: {str(e)}")
            except ImportError:
                pass
            
            raise
    
    return wrapper


# ============================================================================
# 測試程式碼
# ============================================================================

if __name__ == "__main__":
    # 測試日誌功能
    logger = get_logger("test_module")
    
    logger.debug("這是 DEBUG 訊息")
    logger.info("這是 INFO 訊息")
    logger.warning("這是 WARNING 訊息")
    logger.error("這是 ERROR 訊息")
    logger.critical("這是 CRITICAL 訊息")
    
    # 測試例外記錄
    try:
        raise ValueError("這是一個測試例外")
    except Exception as e:
        log_error(logger, e, "測試例外處理")
    
    # 測試效能記錄
    import time
    start = time.time()
    time.sleep(0.1)
    AppLogger.log_performance(logger, "測試操作", time.time() - start)
    
    print(f"\n✅ 日誌已記錄到: {AppLogger._log_file}")
