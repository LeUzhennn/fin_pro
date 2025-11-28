import streamlit as st

def generate_shap_summary(shap_values, features_df, top_n=3):
    """
    根據 SHAP 值生成自然語言摘要 (不使用 Pandas DataFrame 以提高穩健性)。
    """
    try:
        feature_names = features_df.columns.tolist()
        feature_values = features_df.iloc[0].values

        # 為了絕對安全，將特徵和 SHAP 值對齊到最短的長度
        contributors = []
        min_len = min(len(feature_names), len(shap_values))
        for i in range(min_len):
            contributors.append({
                'feature': feature_names[i],
                'value': feature_values[i],
                'shap_value': shap_values[i]
            })

        # 排序以取得貢獻最大的特徵
        contributors.sort(key=lambda x: x['shap_value'], reverse=True)

        # 取得正向貢獻的特徵
        positive_contributors = [c for c in contributors if c['shap_value'] > 0][:top_n]

        # 取得負向貢獻的特徵
        negative_contributors = sorted(
            [c for c in contributors if c['shap_value'] < 0],
            key=lambda x: x['shap_value']
        )[:top_n]

        summary = "#### 📖 簡易分析摘要\n"
        summary += "此預測結果主要基於以下關鍵證據：\n\n"

        if positive_contributors:
            summary += "**主要推力 (判斷為攻擊):**\n"
            for item in positive_contributors:
                summary += f"- **{item['feature']}** 的值為 **{item['value']:.2f}**，顯著地增加了攻擊的可能性。\n"
            summary += "\n"

        if negative_contributors:
            summary += "**反向拉力 (傾向判斷為正常):**\n"
            for item in negative_contributors:
                summary += f"- **{item['feature']}** 的值為 **{item['value']:.2f}**，降低了攻擊的可能性。\n"

        # 如果都沒有，顯示一個通用訊息
        if not positive_contributors and not negative_contributors:
            summary += "無法確定影響預測結果的關鍵特徵。\n"
            
        return summary
    except Exception as e:
        return f"#### 📖 簡易分析摘要\n無法產生分析摘要，錯誤：`{e}`\n"
