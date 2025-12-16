# 🚀 快速入門指南

本專案已完成重構，提升了程式碼品質、可維護性和效能。

## 📦 安裝依賴

```bash
pip install -r requirements.txt
```

## 🎮 啟動應用程式

```bash
streamlit run app.py
```

## 🔧 配置管理

### 修改配置
編輯 `config.py` 來調整各項設定：

```python
# 修改模型參數
class ModelConfig:
    RF_N_ESTIMATORS = 200  # 隨機森林樹的數量
    TEST_SIZE = 0.3        # 測試集比例

# 修改快取策略
class CacheConfig:
    CACHE_TTL = 7200       # 快取 2 小時
    MAX_CACHE_ENTRIES = 20  # 最多 20 個快取條目
```

### 驗證配置
```bash
python config.py
```

## 📝 日誌管理

### 查看日誌
日誌會自動儲存在 `logs/` 目錄：
```bash
tail -f logs/app_20251216.log
```

### 在程式碼中使用日誌
```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("這是資訊訊息")
logger.warning("這是警告訊息")
logger.error("這是錯誤訊息")
```

## 🧪 測試模組

### 測試配置檔案
```bash
python config.py
```

### 測試日誌系統
```bash
python -m src.logger
```

### 測試資料預處理
```bash
python -m src.data_preprocessor
```

## 📂 目錄結構

```
fin_pro_1128/
├── config.py              # 配置管理（新增）
├── app.py                 # 主程式（已更新）
├── requirements.txt       # 依賴套件
├── REFACTORING_SUMMARY.md # 重構總結（新增）
├── QUICKSTART.md          # 本檔案（新增）
│
├── data/                  # 資料目錄
│   └── 03-01-2018.csv
│
├── logs/                  # 日誌目錄（自動建立）
│   └── app_*.log
│
├── models/                # 模型目錄（自動建立）
│
├── plots/                 # 圖表目錄
│
├── src/                   # 核心模組
│   ├── logger.py          # 日誌管理（新增）
│   ├── data_preprocessor.py  # 預處理共用（新增）
│   ├── data_loader.py     # 資料載入（已重構）
│   ├── feature_selector.py   # 特徵選擇（已修正）
│   ├── model_trainer.py   # 模型訓練（已更新）
│   ├── data_analyzer.py   # 資料分析（已更新）
│   ├── build_knowledge_base.py  # RAG 知識庫（已更新）
│   └── llm_analyzer.py    # LLM 分析
│
└── ui/                    # UI 模組
    ├── sidebar.py         # 側邊欄（已更新）
    ├── tab_dashboard.py   # 儀表板
    ├── tab_single_prediction.py  # 單筆預測（已更新）
    ├── tab_batch_prediction.py   # 批次預測
    └── utils.py           # 工具函數（已更新）
```

## 🎯 主要改進

### 1. 配置管理
- ✅ 所有硬編碼參數移至 `config.py`
- ✅ 支援環境變數
- ✅ 配置驗證功能

### 2. 日誌系統
- ✅ 統一的日誌格式
- ✅ 同時輸出到控制台和檔案
- ✅ 自動建立帶時間戳記的日誌

### 3. 程式碼重用
- ✅ 抽取共用的資料預處理函數
- ✅ 消除程式碼重複

### 4. 錯誤處理
- ✅ 修正 DEAP 重複註冊問題
- ✅ 統一的錯誤記錄方式

### 5. 效能優化
- ✅ 快取策略改進（TTL + max_entries）
- ✅ 防止記憶體溢出

## 💡 使用範例

### 使用資料預處理器
```python
from src.data_preprocessor import preprocess_input_for_model

# 預處理使用者輸入
processed_data = preprocess_input_for_model(
    user_input_df,
    scaler,
    selected_features
)

# 進行預測
predictions = model.predict(processed_data)
```

### 使用日誌記錄錯誤
```python
from src.logger import get_logger, log_error

logger = get_logger(__name__)

try:
    # 你的程式碼
    risky_operation()
except Exception as e:
    log_error(logger, e, "執行危險操作時發生錯誤")
    raise
```

### 載入並清理資料
```python
from src.data_preprocessor import load_and_preprocess_data

# 完整的載入和預處理流程
df = load_and_preprocess_data(
    file_path="data/03-01-2018.csv",
    clean_strategy="drop"  # 或 "fill_median", "fill_zero"
)
```

## 🐛 除錯技巧

### 1. 查看詳細日誌
```bash
# 即時查看日誌
tail -f logs/app_*.log

# 搜尋錯誤訊息
grep ERROR logs/app_*.log

# 查看特定模組的日誌
grep "src.feature_selector" logs/app_*.log
```

### 2. 驗證配置
```bash
# 執行配置驗證
python config.py

# 檢查配置值
python -c "from config import ModelConfig; print(ModelConfig.RF_N_ESTIMATORS)"
```

### 3. 測試個別模組
```bash
# 測試資料預處理
python -m src.data_preprocessor

# 測試日誌系統
python -m src.logger
```

## 📚 更多資訊

- **完整重構說明**: 請參閱 `REFACTORING_SUMMARY.md`
- **配置選項**: 請參閱 `config.py` 中的註解
- **API 文件**: 每個函數都有完整的 docstring

## 🤝 貢獻指南

1. 修改配置時，請同時更新 `config.py` 的註解
2. 新增功能時，請使用日誌記錄關鍵操作
3. 提交前請執行配置驗證: `python config.py`
4. 遵循現有的程式碼風格和型別提示

## 📞 問題回報

如遇到問題，請檢查：
1. `logs/` 目錄中的日誌檔案
2. 是否正確設定 `config.py`
3. 所有依賴套件是否已安裝

---

最後更新: 2025-12-16
版本: 1.0
