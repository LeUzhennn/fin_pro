"""
此模組負責資料的讀取、解析與初步清理。
遵循單一職責原則，將資料處理和 UI 顯示分離。
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
from typing import Optional, Tuple

from config import CacheConfig
from src.logger import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=CacheConfig.CACHE_TTL, max_entries=CacheConfig.MAX_CACHE_ENTRIES)
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    從指定路徑載入 CSV 檔案。
    
    此函數使用 Streamlit 快取機制，避免重複載入相同檔案。
    快取會在 1 小時後過期，且最多保留 10 個快取條目。

    Args:
        file_path: CSV 檔案的路徑

    Returns:
        Optional[pd.DataFrame]: 載入的資料，如果載入失敗則回傳 None
    """
    if not os.path.exists(file_path):
        logger.error(f"找不到檔案: {file_path}")
        return None
    
    try:
        logger.info(f"正在載入檔案: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"載入成功: {df.shape[0]} 筆資料, {df.shape[1]} 個欄位")
        return df
    except Exception as e:
        logger.error(f"讀取檔案時發生錯誤: {e}")
        return None


def clean_data_core(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    清理 DataFrame，處理無窮值和遺失值（核心邏輯，不包含 UI）。
    
    處理步驟：
    1. 將無窮值替換為 NaN
    2. 移除包含 NaN 的資料列
    3. 回傳清理後的資料和統計資訊

    Args:
        df: 原始 DataFrame

    Returns:
        Tuple[pd.DataFrame, dict]: (清理後的 DataFrame, 統計資訊字典)
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    
    # 將無窮值替換為 NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 計算 NaN 數量
    nan_rows = df_clean.isnull().any(axis=1).sum()
    
    # 移除包含 NaN 的資料列
    if nan_rows > 0:
        df_clean.dropna(inplace=True)
    
    cleaned_rows = len(df_clean)
    removed_rows = original_rows - cleaned_rows
    
    # 統計資訊
    stats = {
        'original_rows': original_rows,
        'cleaned_rows': cleaned_rows,
        'removed_rows': removed_rows,
        'nan_rows': nan_rows,
        'removal_rate': (removed_rows / original_rows * 100) if original_rows > 0 else 0
    }
    
    logger.info(f"資料清理完成: 原始 {original_rows} 筆 -> 剩餘 {cleaned_rows} 筆 (移除 {removed_rows} 筆)")
    
    return df_clean, stats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理 DataFrame 並在 Streamlit UI 中顯示過程（包含 UI 顯示）。
    
    這是提供給 Streamlit 應用程式使用的函數，會在介面上顯示清理資訊。

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 清理後的 DataFrame
    """
    st.header("資料清理")
    
    df_clean, stats = clean_data_core(df)
    
    # 顯示清理結果
    if stats['nan_rows'] > 0:
        st.info(f"發現 {stats['nan_rows']} 筆包含無效值(NaN/Infinity)的資料，將予以移除。")
    else:
        st.success("資料完整，沒有發現無效值(NaN/Infinity)。")
    
    # 顯示統計資訊
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("原始資料", f"{stats['original_rows']:,} 筆")
    with col2:
        st.metric("有效資料", f"{stats['cleaned_rows']:,} 筆")
    with col3:
        st.metric("移除比例", f"{stats['removal_rate']:.1f}%")
    
    return df_clean


def load_and_display_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    載入並顯示資料的完整流程（包含 UI 顯示）。
    
    這是一個便利函數，結合了載入和清理的完整流程。

    Args:
        file_path: CSV 檔案的路徑

    Returns:
        Optional[pd.DataFrame]: 清理後的資料，如果失敗則回傳 None
    """
    # 載入資料
    df = load_data(file_path)
    
    if df is None:
        st.error(f"❌ 無法載入檔案: {file_path}")
        return None
    
    # 顯示基本資訊
    st.success(f"✅ 資料載入成功!")
    
    # 清理資料
    df_clean = clean_data(df)
    
    return df_clean