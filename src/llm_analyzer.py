# src/llm_analyzer.py

import streamlit as st

def get_threat_explanation(features: dict, label: str) -> str:
    """
    Minimal, import-safe function.
    Output:
      - Title line
      - Suspicious features (only non-zero / non-empty)
    No '建議處理' section.
    """
    # 若你在別處已確保 token，不需要也可移除此段
    _ = st.secrets.get("HUGGINGFACE_API_TOKEN", None)

    # 只列出「有資訊量」的特徵：排除 0 / 空值 / N/A
    lines = []
    for k, v in (features or {}).items():
        if v is None:
            continue
        s = str(v).strip()
        if s.lower() in {"", "n/a", "na", "nan", "null", "none"}:
            continue
        try:
            if float(s) == 0.0:
                continue
        except Exception:
            pass

        lines.append(f"- {k}: {s}")

    if not lines:
        return f"此流量為「{label}」行為\n可疑特徵：目前無可用的非零特徵"

    return f"此流量為「{label}」行為\n可疑特徵：\n" + "\n".join(lines[:8])


if __name__ == "__main__":
    pass
