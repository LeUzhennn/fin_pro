import streamlit as st
import numpy as np
import requests
import io
from typing import Optional

from config import UIConfig
from src.logger import get_logger

logger = get_logger(__name__)

def generate_shap_summary(
    shap_values,
    features_df,
    predicted_label,
    le,
    shap_base_value,
    top_n: Optional[int] = None
):
    """
    根據 SHAP 值生成完全透明的自然語言摘要，包含影響力分數和最終判斷公式。
    
    Args:
        shap_values: SHAP 值陣列
        features_df: 特徵 DataFrame
        predicted_label: 預測的標籤
        le: LabelEncoder 實例
        shap_base_value: SHAP 基礎值
        top_n: 顯示前 N 個特徵，預設使用配置檔案設定
        
    Returns:
        str: 格式化的 SHAP 分析摘要
    """
    if top_n is None:
        top_n = UIConfig.SHAP_TOP_N_FEATURES
    
    try:
        # 確定 "另一方" 的標籤
        all_classes = le.classes_.tolist()
        other_label = next((c for c in all_classes if c != predicted_label), "另一類別")

        feature_names = features_df.columns.tolist()
        feature_values = features_df.iloc[0].values

        contributors = []
        min_len = min(len(feature_names), len(shap_values))
        for i in range(min_len):
            contributors.append({
                'feature': feature_names[i],
                'value': feature_values[i],
                'shap_value': shap_values[i]
            })

        # SHAP 值 > 0，表示支持當前預測結果的證據
        positive_contributors = sorted(
            [c for c in contributors if c['shap_value'] > 0],
            key=lambda x: x['shap_value'],
            reverse=True
        )[:top_n]

        # SHAP 值 < 0，表示反對當前預測結果 (即支持另一方) 的證據
        negative_contributors = sorted(
            [c for c in contributors if c['shap_value'] < 0],
            key=lambda x: x['shap_value']
        )[:top_n]

        # --- 開始產生摘要 ---
        summary = f"#### 📖 簡易分析摘要 (最終判定： **{predicted_label}**)\n"
        summary += "模型是這樣權衡正反兩方的證據，才做出最終判斷的：\n\n"

        if positive_contributors:
            summary += f"**主要依據 (判斷為「{predicted_label}」):**\n"
            for item in positive_contributors:
                summary += f"- 當 **{item['feature']}** 的值為 **{item['value']:.2f}** 時，成為一個關鍵指標 (影響力: **+{item['shap_value']:.2f}**)。\n"
            summary += "\n"

        if negative_contributors:
            summary += f"**反向依據 (傾向判斷為「{other_label}」):**\n"
            for item in negative_contributors:
                summary += f"- 當 **{item['feature']}** 的值為 **{item['value']:.2f}** 時，成為一個反向指標 (影響力: **{item['shap_value']:.2f}**)。\n"
            summary += "\n"

        # --- 新增最終判斷公式 ---
        total_shap_score = np.sum(shap_values)
        final_score = shap_base_value + total_shap_score
        
        summary += "--- \n"
        summary += "#### ⚖️ 模型最終判斷公式\n"
        summary += "模型透過一個類似計分的方式來做出最後決定：\n"
        summary += f"- **基礎分數**: **{shap_base_value:.2f}** (註：此為模型對所有訓練資料的平均預測，可視為\"起跑線\")\n"
        summary += f"- **所有特徵總影響力**: **{total_shap_score:+.2f}**\n"
        summary += f"- **最終分數 (基礎分數 + 總影響力)**: **{final_score:.2f}**\n\n"
        
        # Epsilon for float comparison
        epsilon = 1e-6

        if total_shap_score > epsilon:
            summary += f"**判斷依據**：因為所有特徵的**總影響力為正數**，將分數從 {shap_base_value:.2f} **拉高**至 {final_score:.2f}，因此模型最終判斷為「**{predicted_label}**」。\n"
        elif total_shap_score < -epsilon:
            # This case might seem counter-intuitive (e.g. final prediction is "Attack" but score was lowered),
            # but it's possible if the base_score was very high to begin with. The text remains correct.
            summary += f"**判斷依據**：雖然所有特徵的**總影響力為負數**，將分數從 {shap_base_value:.2f} **拉低**至 {final_score:.2f}，但最終分數仍足夠高，因此模型依然判斷為「**{predicted_label}**」。\n"
        else: # total_shap_score is effectively zero
            summary += f"**判斷依據**：因為所有特徵的正反向影響力幾乎完全互相抵消 (總影響力 ≈ 0)，所以模型的判斷主要回歸到**基礎分數**。由於基礎分數 ({shap_base_value:.2f}) 本身就傾向於「**{predicted_label}**」，因此這是最終的預測結果。\n"
            
        return summary
    except Exception as e:
        return f"#### 📖 簡易分析摘要\n無法產生分析摘要，錯誤：`{e}`\n"


def download_file_from_gdrive(url):
    """
    Downloads a file from a Google Drive URL, handling the large file confirmation prompt.
    Returns the file content in bytes.
    """
    session = requests.Session()
    response = session.get(url, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        url_with_confirm = url + '&confirm=' + token
        response = session.get(url_with_confirm, stream=True)

    if 'Content-Disposition' not in response.headers:
        error_html = response.text
        if 'Google Drive' in error_html and 'virus scan' in error_html:
            raise Exception("下載失敗：無法自動繞過 Google Drive 的病毒掃描警告。")
        else:
            raise Exception("下載失敗：回應不是一個檔案，而是一個 HTML 頁面。請檢查 URL 和共用權限。")

    file_buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=32768):
        if chunk:
            file_buffer.write(chunk)

    return file_buffer.getvalue()
