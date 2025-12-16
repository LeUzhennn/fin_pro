"""
配置管理模組
集中管理所有應用程式的配置項目、路徑和環境變數
"""
import os
from pathlib import Path
from typing import Optional

# ============================================================================
# 基礎路徑配置
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "db"

# ============================================================================
# 資料檔案配置
# ============================================================================
DEFAULT_DATA_FILE = DATA_DIR / "03-01-2018.csv"
TEST_DATA_FILE = "test.csv"

# ============================================================================
# 模型訓練配置
# ============================================================================
class ModelConfig:
    """模型訓練相關配置"""
    # 隨機森林參數
    RF_N_ESTIMATORS = 100
    RF_RANDOM_STATE = 42
    RF_N_JOBS = -1
    
    # 資料分割參數
    TEST_SIZE = 0.2
    TRAIN_TEST_RANDOM_STATE = 42
    
    # 交叉驗證參數
    CV_FOLDS = 3
    CV_SCORING = 'accuracy'

# ============================================================================
# 特徵選擇配置
# ============================================================================
class FeatureSelectionConfig:
    """基因演算法特徵選擇配置"""
    # DEAP 基因演算法參數
    POPULATION_SIZE = 40
    NUM_GENERATIONS = 15
    CROSSOVER_PROB = 0.5
    MUTATION_PROB = 0.2
    MUTATION_INDPB = 0.05
    TOURNAMENT_SIZE = 3
    
    # 評估用的隨機森林參數
    GA_RF_N_ESTIMATORS = 20
    GA_RF_RANDOM_STATE = 42
    GA_RF_N_JOBS = -1

# ============================================================================
# 資料處理配置
# ============================================================================
class DataConfig:
    """資料處理相關配置"""
    # 視覺化特徵列表
    SELECTED_FEATURES_FOR_VISUALIZATION = [
        'Flow Duration', 
        'Flow IAT Mean', 
        'Tot Fwd Pkts'
    ]
    
    # 資料清理參數
    MAX_LABELS_FOR_PLOT = 5
    
    # 文字分隔符號
    INFINITY_STR = 'Infinity'
    NEG_INFINITY_STR = '-Infinity'

# ============================================================================
# UI 配置
# ============================================================================
class UIConfig:
    """使用者介面配置"""
    PAGE_TITLE = "入侵偵測互動式分析系統"
    PAGE_ICON = "🛡️"
    LAYOUT = "wide"
    
    # SHAP 分析參數
    SHAP_TOP_N_FEATURES = 3
    
    # 表單欄位數量
    FORM_NUM_COLUMNS = 4

# ============================================================================
# 快取配置
# ============================================================================
class CacheConfig:
    """快取策略配置"""
    # 快取時間 (秒)
    CACHE_TTL = 3600  # 1 小時
    
    # 最大快取條目數
    MAX_CACHE_ENTRIES = 10
    
    # 是否顯示快取載入提示
    SHOW_SPINNER = False

# ============================================================================
# API 與認證配置
# ============================================================================
class APIConfig:
    """API 相關配置"""
    # HuggingFace API Token (從環境變數或 secrets 讀取)
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # RAG 相關配置
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # PDF 處理配置
    PDF_DIRECTORY = str(BASE_DIR)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

# ============================================================================
# 日誌配置
# ============================================================================
class LogConfig:
    """日誌系統配置"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# 確保必要的目錄存在
# ============================================================================
def ensure_directories():
    """建立必要的目錄結構"""
    for directory in [DATA_DIR, PLOTS_DIR, MODEL_DIR, DB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# 啟動時自動建立目錄
ensure_directories()

# ============================================================================
# 配置驗證
# ============================================================================
def validate_config() -> tuple[bool, list[str]]:
    """
    驗證配置的完整性和正確性
    
    Returns:
        tuple[bool, list[str]]: (是否通過驗證, 錯誤訊息列表)
    """
    errors = []
    
    # 檢查關鍵路徑是否存在
    if not DATA_DIR.exists():
        errors.append(f"資料目錄不存在: {DATA_DIR}")
    
    if not DEFAULT_DATA_FILE.exists():
        errors.append(f"預設資料檔案不存在: {DEFAULT_DATA_FILE}")
    
    # 檢查模型參數
    if ModelConfig.RF_N_ESTIMATORS <= 0:
        errors.append("隨機森林樹的數量必須大於 0")
    
    if not 0 < ModelConfig.TEST_SIZE < 1:
        errors.append("測試集比例必須在 0 到 1 之間")
    
    # 檢查特徵選擇參數
    if FeatureSelectionConfig.POPULATION_SIZE <= 0:
        errors.append("族群大小必須大於 0")
    
    if FeatureSelectionConfig.NUM_GENERATIONS <= 0:
        errors.append("世代數必須大於 0")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # 執行配置驗證
    is_valid, errors = validate_config()
    if is_valid:
        print("✅ 配置驗證通過")
        print(f"📁 基礎目錄: {BASE_DIR}")
        print(f"📁 資料目錄: {DATA_DIR}")
        print(f"📁 模型目錄: {MODEL_DIR}")
        print(f"📁 圖表目錄: {PLOTS_DIR}")
    else:
        print("❌ 配置驗證失敗:")
        for error in errors:
            print(f"  - {error}")
