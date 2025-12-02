# 🛡️ 入侵偵測互動式分析系統

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

這是一個基於 Streamlit 的互動式網頁應用，旨在分析網路入侵偵測資料。使用者可以透過簡單的網頁介面，進行資料探索、特徵篩選、模型訓練、即時預測，並透過 SHAP 可解釋性分析了解模型為何會做出特定判斷。

## ✨ 主要功能

*   **📈 儀表板 & 模型評估**:
    *   視覺化分析資料集。
    *   評估模型效能，包含準確率、精確率、召回率、F1 分數。
    *   透過混淆矩陣深入了解模型在各類別上的表現。

*   **🔬 即時單筆預測**:
    *   手動輸入單筆網路流量特徵，即時獲得模型預測結果。
    *   透過 SHAP Force Plot 視覺化分析，了解單次預測中各特徵的影響力。

*   **🗂️ 批次流量分析**:
    *   上傳 CSV 檔案，對多筆流量資料進行批次預測。
    *   互動式篩選，方便查看正常或異常的流量。
    *   提供下載分析後的結果。

*   **🤖 模型訓練與管理**:
    *   從側邊欄輕鬆載入、清理資料。
    *   進行特徵選擇與模型訓練。
    *   支援從本機或 URL 載入已訓練好的模型。

## 🚀 如何執行

1.  **複製專案庫**
    ```bash
    git clone https://your-repository-url.git
    cd your-project-directory
    ```

2.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```

3.  **啟動 Streamlit 應用**
    ```bash
    streamlit run app.py
    ```
    應用程式將會在您的瀏覽器中開啟。

## 📂 專案結構

```
.
├── 📄 app.py                # Streamlit 應用程式主檔案
├── 📄 requirements.txt      # 專案依賴套件
├── 📁 .streamlit/
│   └── 📄 config.toml       # Streamlit 設定檔 (例如：最大上傳大小)
├── 📁 data/
│   └── 📄 03-01-2018.csv    # 範例資料集
├── 📁 src/
│   ├── 📄 data_loader.py    # 資料讀取模組
│   ├── 📄 feature_selector.py # 特徵選擇模組
│   └── 📄 model_trainer.py  # 模型訓練模組
└── 📁 ui/
    ├── 📄 sidebar.py        # 側邊欄介面
    ├── 📄 tab_dashboard.py    # 儀表板分頁
    ├── 📄 tab_single_prediction.py # 即時預測分頁
    ├── 📄 tab_batch_prediction.py  # 批次分析分頁
    └── 📄 utils.py          # 共用工具函式
```

## 🛠️ 技術堆疊

*   **程式語言:** Python 3.9+
*   **Web 框架:** Streamlit
*   **資料處理:** Pandas, Numpy
*   **機器學習:** Scikit-learn, `sklearn-genetic-opt`
*   **模型可解釋性:** SHAP
*   **資料視覺化:** Matplotlib, Seaborn

## 📜 開發日誌 (精簡)

*   **2025-11-18**: 重構批次分析功能，優化 UI/UX，並修復核心錯誤。
*   **2025-11-18**: 新增混淆矩陣視覺化，並整合 SHAP 可解釋性分析。
*   **2025-11-22**: 將介面重構成多分頁結構，提升程式碼模組化與穩定性。
*   **2025-11-28**: 調整 Streamlit 上傳大小限制，並持續優化 SHAP 分析結果的文字說明，使其更易於理解。

詳細的開發紀錄與規格請參閱 `openspec/project.md`。
