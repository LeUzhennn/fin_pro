# AI 威脅分析升級指南

## 📊 當前實作（v1.1.0）

✅ **基於規則的智能分析**
- 使用領域知識和專家規則
- 6 種常見攻擊類型的詳細分析
- 即時、無延遲、免費使用
- 不需要外部 API

---

## 🚀 進階升級選項

如果需要更強大的 AI 分析能力，可以選擇以下方案：

### 選項 1: 整合 HuggingFace Inference API（免費）

**優點**:
- 真正的 LLM 生成分析
- 免費使用額度
- 支援多種開源模型

**實作步驟**:

1. **取得 API Token**
   ```bash
   # 訪問 https://huggingface.co/settings/tokens
   # 建立新的 Access Token
   ```

2. **設定環境變數**
   ```bash
   # 在 .streamlit/secrets.toml 中加入
   HUGGINGFACE_API_TOKEN = "hf_xxxxxxxxxxxxx"
   ```

3. **更新程式碼**
   ```python
   import requests
   
   def call_llm_api(prompt: str, token: str) -> str:
       API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
       
       headers = {"Authorization": f"Bearer {token}"}
       payload = {
           "inputs": prompt,
           "parameters": {
               "max_new_tokens": 500,
               "temperature": 0.7
           }
       }
       
       response = requests.post(API_URL, headers=headers, json=payload)
       return response.json()[0]['generated_text']
   ```

---

### 選項 2: 使用本地 RAG 知識庫

**優點**:
- 基於您的 PDF 文件
- 完全本地運行
- 專業領域知識

**實作步驟**:

1. **確保已建立知識庫**
   ```bash
   python -m src.build_knowledge_base
   ```

2. **整合到 llm_analyzer.py**
   ```python
   from langchain_community.vectorstores import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings
   
   def query_knowledge_base(question: str) -> str:
       embeddings = HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2"
       )
       vectordb = Chroma(
           persist_directory="db",
           embedding_function=embeddings
       )
       
       docs = vectordb.similarity_search(question, k=3)
       return "\n".join([doc.page_content for doc in docs])
   ```

---

### 選項 3: 整合 OpenAI API（付費）

**優點**:
- 最強大的分析能力
- 高品質的自然語言輸出
- 可自訂 prompt

**實作步驟**:

1. **安裝套件**
   ```bash
   pip install openai
   ```

2. **設定 API Key**
   ```bash
   # 在 .streamlit/secrets.toml 中
   OPENAI_API_KEY = "sk-xxxxxxxxxxxxx"
   ```

3. **程式碼範例**
   ```python
   import openai
   
   def analyze_with_gpt(features: dict, label: str) -> str:
       prompt = f"""
       作為資安專家，請分析以下網路流量特徵：
       
       攻擊類型: {label}
       特徵數據: {features}
       
       請提供：
       1. 攻擊行為分析
       2. 風險評估
       3. 具體處理建議
       """
       
       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[
               {"role": "system", "content": "你是一位專業的網路資安分析師"},
               {"role": "user", "content": prompt}
           ],
           temperature=0.7,
           max_tokens=800
       )
       
       return response.choices[0].message.content
   ```

---

## 🎨 混合式方案（推薦）

結合規則和 AI 的優點：

```python
def get_threat_explanation_hybrid(features: dict, label: str) -> str:
    """混合式威脅分析"""
    
    # 1. 使用規則快速生成基礎分析
    base_analysis = get_threat_explanation_rule_based(features, label)
    
    # 2. 如果有 API Token，增強分析
    if has_llm_api():
        ai_insight = call_llm_for_insight(features, label)
        return f"{base_analysis}\n\n### 🤖 AI 深度分析\n{ai_insight}"
    
    return base_analysis
```

---

## 📈 效能對比

| 方案 | 速度 | 成本 | 品質 | 離線可用 |
|------|------|------|------|----------|
| **規則分析（目前）** | ⚡ 即時 | 💰 免費 | ⭐⭐⭐⭐ | ✅ 是 |
| HuggingFace API | 🐌 2-5秒 | 💰 免費 | ⭐⭐⭐⭐ | ❌ 否 |
| 本地 RAG | ⚡ 快速 | 💰 免費 | ⭐⭐⭐⭐⭐ | ✅ 是 |
| OpenAI GPT-4 | 🐌 3-8秒 | 💵 付費 | ⭐⭐⭐⭐⭐ | ❌ 否 |
| 混合式 | ⚡ 快速 | 💰 免費/付費 | ⭐⭐⭐⭐⭐ | ⚠️ 部分 |

---

## 🛠️ 快速實作：混合式方案

建立 `src/llm_analyzer_enhanced.py`:

```python
"""增強版 AI 威脅分析（可選）"""

import streamlit as st
from src.llm_analyzer import get_threat_explanation
from src.logger import get_logger

logger = get_logger(__name__)


def get_threat_explanation_enhanced(features: dict, label: str) -> str:
    """
    增強版威脅分析
    
    優先使用規則分析，如果配置了 LLM API 則增加 AI 洞察
    """
    # 基礎分析（規則）
    base_report = get_threat_explanation(features, label)
    
    # 檢查是否有 API 配置
    try:
        api_token = st.secrets.get("HUGGINGFACE_API_TOKEN")
        
        if api_token:
            logger.info("偵測到 LLM API，啟用增強分析")
            
            # 建立 prompt
            prompt = f"""你是資安專家。分析這個{label}攻擊，特徵: {list(features.keys())[:5]}。
用繁體中文提供簡短的專業建議（100字內）。"""
            
            try:
                ai_insight = call_huggingface_api(prompt, api_token)
                
                if ai_insight:
                    return f"{base_report}\n\n### 🤖 AI 專家洞察\n{ai_insight}"
            except Exception as e:
                logger.warning(f"LLM API 呼叫失敗，使用基礎分析: {e}")
    
    except Exception as e:
        logger.debug(f"未配置 LLM API: {e}")
    
    # 回傳基礎分析
    return base_report


def call_huggingface_api(prompt: str, token: str, timeout: int = 10) -> str:
    """呼叫 HuggingFace Inference API"""
    import requests
    
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }
    
    try:
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
    except:
        pass
    
    return ""
```

---

## 🎯 建議選擇

### 現在就可用（已實作）
✅ **規則式分析** - 專業、快速、免費、離線可用

### 想要更強大
🚀 **本地 RAG** - 基於您的文件，完全客製化

### 需要最佳效果
💎 **混合式方案** - 規則 + LLM API，平衡速度和品質

---

## 📞 需要幫助？

如果想實作進階方案，請告訴我您想要：
1. HuggingFace API 整合（免費）
2. 本地 RAG 整合（已有基礎）
3. OpenAI API 整合（付費但最強）
4. 混合式方案（推薦）

---

更新日期: 2025-12-16
版本: 1.1.0
