"""
資料預處理共用模組
抽取重複的資料預處理邏輯，提供統一的介面
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

from config import DataConfig
from src.logger import get_logger

logger = get_logger(__name__)


def preprocess_input_for_model(
    user_input_df: pd.DataFrame,
    scaler: StandardScaler,
    selected_features: List[str]
) -> pd.DataFrame:
    """
    統一的輸入資料預處理流程，用於單筆或批次預測
    
    此函數處理以下步驟：
    1. 建立完整的特徵 DataFrame（填補缺失的特徵為 0）
    2. 使用訓練時的 scaler 進行標準化
    3. 選取模型所需的特徵子集
    
    Args:
        user_input_df: 使用者輸入的原始資料 DataFrame
        scaler: 訓練時使用的 StandardScaler
        selected_features: 模型訓練時選擇的特徵列表
        
    Returns:
        pd.DataFrame: 預處理後、可直接餵入模型的 DataFrame
        
    Example:
        >>> input_data = pd.DataFrame([{'feature1': 1.0, 'feature2': 2.0}])
        >>> processed_data = preprocess_input_for_model(input_data, scaler, ['feature1', 'feature2'])
    """
    try:
        # 取得 scaler 訓練時的完整特徵列表
        required_features = scaler.feature_names_in_
        
        # 建立完整的特徵 DataFrame，缺失的特徵用 0 填補
        full_df = pd.DataFrame(0.0, index=user_input_df.index, columns=required_features)
        
        # 填入使用者提供的特徵值
        for col in user_input_df.columns:
            if col in full_df.columns:
                full_df[col] = user_input_df[col]
        
        # 標準化
        scaled_data = scaler.transform(full_df)
        scaled_df = pd.DataFrame(scaled_data, index=full_df.index, columns=required_features)
        
        # 選取模型所需的特徵
        final_df = scaled_df[selected_features]
        
        logger.debug(f"預處理完成: 輸入 {user_input_df.shape} -> 輸出 {final_df.shape}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"資料預處理失敗: {e}")
        raise


def clean_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = 'drop'
) -> pd.DataFrame:
    """
    清理 DataFrame 中的數值欄位，處理無窮值和 NaN
    
    Args:
        df: 要清理的 DataFrame
        columns: 要清理的欄位列表，若為 None 則處理所有數值欄位
        strategy: 處理策略 ('drop': 刪除, 'fill_median': 填補中位數, 'fill_zero': 填 0)
        
    Returns:
        pd.DataFrame: 清理後的 DataFrame
    """
    df_copy = df.copy()
    
    # 如果沒有指定欄位，則處理所有數值欄位
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # 替換無窮值為 NaN
    df_copy[columns] = df_copy[columns].replace([np.inf, -np.inf], np.nan)
    
    # 根據策略處理 NaN
    if strategy == 'drop':
        original_rows = len(df_copy)
        df_copy = df_copy.dropna(subset=columns)
        removed_rows = original_rows - len(df_copy)
        if removed_rows > 0:
            logger.info(f"刪除了 {removed_rows} 筆包含無效值的資料")
    
    elif strategy == 'fill_median':
        for col in columns:
            if df_copy[col].isnull().any():
                median_value = df_copy[col].median()
                df_copy[col] = df_copy[col].fillna(median_value)
                logger.debug(f"欄位 '{col}' 使用中位數 {median_value} 填補")
    
    elif strategy == 'fill_zero':
        df_copy[columns] = df_copy[columns].fillna(0)
        logger.debug(f"欄位 {columns} 使用 0 填補")
    
    else:
        raise ValueError(f"不支援的策略: {strategy}")
    
    return df_copy


def convert_to_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    errors: str = 'coerce'
) -> pd.DataFrame:
    """
    將 DataFrame 中的欄位轉換為數值型別
    
    Args:
        df: 要轉換的 DataFrame
        columns: 要轉換的欄位列表，若為 None 則處理所有非數值欄位
        errors: 轉換錯誤的處理方式 ('raise', 'coerce', 'ignore')
        
    Returns:
        pd.DataFrame: 轉換後的 DataFrame
    """
    df_copy = df.copy()
    
    # 如果沒有指定欄位，則處理所有物件型別欄位
    if columns is None:
        columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        # 排除明顯的非數值欄位
        columns = [col for col in columns if col not in ['Label', 'Timestamp']]
    
    for col in columns:
        if col in df_copy.columns:
            # 移除逗號和特殊字串
            df_copy[col] = (
                df_copy[col]
                .astype(str)
                .str.replace(',', '')
                .replace(DataConfig.INFINITY_STR, np.nan)
                .replace(DataConfig.NEG_INFINITY_STR, np.nan)
            )
            
            # 轉換為數值
            df_copy[col] = pd.to_numeric(df_copy[col], errors=errors)
            logger.debug(f"欄位 '{col}' 已轉換為數值型別")
    
    return df_copy


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除 DataFrame 欄位名稱的前後空白
    
    Args:
        df: 要處理的 DataFrame
        
    Returns:
        pd.DataFrame: 欄位名稱已清理的 DataFrame
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip()
    
    # 如果有 Label 欄位，也清理其值
    if 'Label' in df_copy.columns:
        df_copy['Label'] = df_copy['Label'].str.strip()
    
    return df_copy


def validate_features(
    df: pd.DataFrame,
    required_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    驗證 DataFrame 是否包含所需的特徵
    
    Args:
        df: 要驗證的 DataFrame
        required_features: 必要的特徵列表
        
    Returns:
        Tuple[bool, List[str]]: (是否通過驗證, 缺失的特徵列表)
    """
    df_columns = set(df.columns)
    required_set = set(required_features)
    missing_features = list(required_set - df_columns)
    
    is_valid = len(missing_features) == 0
    
    if not is_valid:
        logger.warning(f"缺少以下特徵: {missing_features}")
    
    return is_valid, missing_features


def get_feature_statistics(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    計算特徵的統計資訊
    
    Args:
        df: 資料 DataFrame
        features: 要統計的特徵列表
        
    Returns:
        pd.DataFrame: 包含統計資訊的 DataFrame
    """
    stats_df = df[features].describe().T
    stats_df['missing_count'] = df[features].isnull().sum()
    stats_df['missing_percent'] = (stats_df['missing_count'] / len(df) * 100).round(2)
    
    return stats_df


# ============================================================================
# 資料載入相關函數
# ============================================================================

def load_and_preprocess_data(
    file_path: str,
    clean_strategy: str = 'drop'
) -> Optional[pd.DataFrame]:
    """
    載入並預處理資料的完整流程
    
    Args:
        file_path: CSV 檔案路徑
        clean_strategy: 清理策略
        
    Returns:
        Optional[pd.DataFrame]: 處理後的 DataFrame，失敗時回傳 None
    """
    try:
        logger.info(f"正在載入資料: {file_path}")
        
        # 載入資料
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"載入成功: {df.shape[0]} 筆資料, {df.shape[1]} 個欄位")
        
        # 清理欄位名稱
        df = strip_column_names(df)
        
        # 轉換數值欄位
        df = convert_to_numeric(df)
        
        # 清理無效值
        df = clean_numeric_columns(df, strategy=clean_strategy)
        
        logger.info(f"預處理完成: 剩餘 {df.shape[0]} 筆有效資料")
        
        return df
        
    except Exception as e:
        logger.error(f"載入或預處理資料失敗: {e}")
        return None


# ============================================================================
# 測試程式碼
# ============================================================================

if __name__ == "__main__":
    # 測試資料預處理功能
    print("測試資料預處理模組...")
    
    # 建立測試資料
    test_df = pd.DataFrame({
        ' Feature1 ': [1.0, 2.0, np.inf, 4.0],
        'Feature2': ['5', '6', '7', 'invalid'],
        'Label': [' Benign ', 'Attack', 'Benign', 'Attack']
    })
    
    print("\n原始資料:")
    print(test_df)
    
    # 測試清理欄位名稱
    test_df = strip_column_names(test_df)
    print("\n清理欄位名稱後:")
    print(test_df)
    
    # 測試轉換為數值
    test_df = convert_to_numeric(test_df)
    print("\n轉換為數值後:")
    print(test_df)
    
    # 測試清理無效值
    test_df = clean_numeric_columns(test_df, strategy='fill_zero')
    print("\n清理無效值後:")
    print(test_df)
