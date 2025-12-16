# src/llm_analyzer.py
"""
AI 威脅分析模組
提供基於領域知識的智能威脅分析和安全建議
"""

import streamlit as st
from typing import Dict, List, Tuple
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 攻擊類型知識庫
# ============================================================================

ATTACK_KNOWLEDGE = {
    "DDoS": {
        "name": "分散式拒絕服務攻擊 (DDoS)",
        "description": "透過大量請求耗盡目標系統資源，使正常用戶無法訪問服務",
        "indicators": [
            "異常高的封包數量 (Tot Fwd Pkts, Tot Bwd Pkts)",
            "短時間內大量連線 (Flow Duration 短但流量大)",
            "高流量速率 (Flow Byts/s, Flow Pkts/s)",
            "重複的源 IP 或目標 IP"
        ],
        "risks": [
            "🔴 服務中斷，影響業務連續性",
            "🔴 伺服器資源耗盡",
            "🟡 可能掩蓋其他攻擊行為",
            "🟡 品牌聲譽受損"
        ],
        "recommendations": [
            "立即啟動 DDoS 防護機制（如 CDN、Anti-DDoS 服務）",
            "限制單一來源的連線速率",
            "使用負載均衡分散流量",
            "監控異常流量模式並自動封鎖",
            "聯繫 ISP 協助過濾惡意流量"
        ]
    },
    "PortScan": {
        "name": "埠掃描攻擊 (Port Scan)",
        "description": "攻擊者掃描目標系統開放的網路埠，尋找可利用的服務漏洞",
        "indicators": [
            "短時間內連接多個不同埠",
            "大量的 SYN 封包但沒有完成三次握手",
            "小流量但高頻率的連線嘗試",
            "異常的埠號順序訪問"
        ],
        "risks": [
            "🟡 暴露系統架構和服務資訊",
            "🟡 為後續攻擊做準備（偵察階段）",
            "🟠 可能發現未修補的漏洞",
            "🟢 本身不造成直接損害"
        ],
        "recommendations": [
            "立即啟用防火牆規則，限制埠掃描行為",
            "關閉不必要的開放埠",
            "使用 IDS/IPS 偵測並封鎖掃描來源",
            "檢查近期是否有異常登入嘗試",
            "更新系統和服務到最新版本"
        ]
    },
    "Infilteration": {
        "name": "滲透攻擊 (Infiltration)",
        "description": "攻擊者嘗試未經授權進入系統，可能利用漏洞或社交工程",
        "indicators": [
            "異常的登入嘗試模式",
            "非標準的協議使用",
            "資料外傳跡象 (大量 Bwd Pkts)",
            "在非工作時間的異常活動"
        ],
        "risks": [
            "🔴 敏感資料可能被竊取",
            "🔴 系統可能被植入後門",
            "🔴 可能建立持久化存取",
            "🟠 可作為跳板攻擊其他系統"
        ],
        "recommendations": [
            "⚠️ 緊急：立即隔離受影響系統",
            "檢查是否有新增的未授權帳號",
            "掃描是否有惡意軟體或後門程式",
            "審查系統日誌，追蹤攻擊來源和範圍",
            "重設所有相關系統的密碼",
            "進行完整的安全評估"
        ]
    },
    "Bot": {
        "name": "殭屍網路 (Botnet)",
        "description": "受感染的設備被遠端控制，用於執行惡意活動",
        "indicators": [
            "定期與外部 C&C 伺服器通訊",
            "異常的網路流量模式",
            "在背景執行可疑的網路活動",
            "參與 DDoS 攻擊"
        ],
        "risks": [
            "🔴 設備被用於攻擊其他目標",
            "🔴 敏感資料可能被竊取",
            "🟠 網路頻寬被消耗",
            "🟠 可能感染其他內網設備"
        ],
        "recommendations": [
            "⚠️ 立即隔離受感染設備",
            "執行完整的惡意軟體掃描和清除",
            "封鎖與已知 C&C 伺服器的通訊",
            "檢查其他設備是否也被感染",
            "更新防毒軟體和作業系統",
            "教育使用者避免點擊可疑連結"
        ]
    },
    "Web Attack": {
        "name": "網頁攻擊 (Web Attack)",
        "description": "針對網頁應用程式的攻擊，如 SQL 注入、XSS、CSRF 等",
        "indicators": [
            "HTTP 請求中包含異常字元或指令",
            "大量的錯誤回應碼 (400, 500)",
            "異常長的 URL 或 POST 資料",
            "針對特定路徑的重複請求"
        ],
        "risks": [
            "🔴 資料庫可能被竊取或修改",
            "🔴 網站內容可能被篡改",
            "🟠 使用者帳號可能被盜用",
            "🟠 可能散播惡意程式碼"
        ],
        "recommendations": [
            "立即檢查 Web 應用程式日誌",
            "啟用 Web Application Firewall (WAF)",
            "驗證所有使用者輸入並進行過濾",
            "使用參數化查詢防止 SQL 注入",
            "實作 HTTPS 和安全標頭",
            "定期進行滲透測試"
        ]
    },
    "Brute Force": {
        "name": "暴力破解攻擊 (Brute Force)",
        "description": "透過大量嘗試不同的密碼組合來破解帳號",
        "indicators": [
            "短時間內大量登入失敗",
            "來自單一 IP 的重複登入嘗試",
            "使用常見密碼字典攻擊",
            "針對多個帳號的嘗試"
        ],
        "risks": [
            "🔴 帳號可能被破解",
            "🟠 合法使用者可能被鎖定",
            "🟠 消耗系統認證資源",
            "🟡 可能觸發帳號鎖定機制"
        ],
        "recommendations": [
            "立即啟用帳號鎖定機制",
            "實作多因素認證 (MFA/2FA)",
            "使用 CAPTCHA 防止自動化攻擊",
            "限制登入嘗試次數和頻率",
            "封鎖攻擊來源 IP",
            "強制使用強密碼政策"
        ]
    }
}


def get_attack_severity(label: str) -> str:
    """取得攻擊嚴重程度"""
    severity_map = {
        "DDoS": "🔴 極高",
        "Infilteration": "🔴 極高",
        "Bot": "🔴 高",
        "Web Attack": "🟠 中高",
        "PortScan": "🟡 中",
        "Brute Force": "🟠 中高"
    }
    return severity_map.get(label, "🟢 低")


def analyze_feature_anomaly(feature_name: str, value: float) -> str:
    """分析特定特徵的異常程度"""
    # 基於領域知識的閾值分析
    if "Duration" in feature_name:
        if value < 1000:
            return "⚡ 極短連線時間（可能為掃描或快速攻擊）"
        elif value > 1000000:
            return "🕐 異常長時間連線（可能為資料竊取或持久化連線）"
    
    elif "Pkts" in feature_name:
        if value > 10000:
            return "📦 封包數量異常龐大（可能為 DDoS 或資料傳輸）"
        elif value > 1000:
            return "📦 封包數量偏高"
    
    elif "IAT" in feature_name:
        if value < 10:
            return "⚡ 封包間隔極短（高速攻擊）"
        elif value > 100000:
            return "🕐 封包間隔極長（慢速攻擊或偵察）"
    
    elif "Byts/s" in feature_name or "Pkts/s" in feature_name:
        if value > 1000000:
            return "🌊 流量速率極高（可能為頻寬消耗攻擊）"
    
    return ""


def get_threat_explanation(features: dict, label: str) -> str:
    """
    進階 AI 威脅分析
    
    提供基於領域知識的專業威脅分析，包含：
    - 攻擊類型說明
    - 關鍵特徵分析
    - 風險評估
    - 專業處理建議
    
    Args:
        features: 流量特徵字典
        label: 預測的攻擊類型標籤
        
    Returns:
        str: Markdown 格式的分析報告
    """
    logger.info(f"開始分析威脅: {label}")
    
    # 取得攻擊類型資訊
    attack_info = ATTACK_KNOWLEDGE.get(label, {
        "name": label,
        "description": "未知攻擊類型",
        "indicators": [],
        "risks": ["需要進一步分析"],
        "recommendations": ["建議諮詢資安專家"]
    })
    
    # 建立分析報告
    report = []
    
    # 標題和嚴重程度
    severity = get_attack_severity(label)
    report.append(f"## 🚨 威脅分析報告")
    report.append(f"")
    report.append(f"**攻擊類型**: {attack_info['name']}")
    report.append(f"**嚴重程度**: {severity}")
    report.append(f"")
    
    # 攻擊說明
    report.append(f"### 📋 攻擊描述")
    report.append(f"{attack_info['description']}")
    report.append(f"")
    
    # 關鍵特徵分析
    report.append(f"### 🔍 關鍵特徵分析")
    
    # 過濾有效特徵
    valid_features = []
    for k, v in (features or {}).items():
        if v is None:
            continue
        try:
            float_val = float(v)
            if float_val != 0.0:
                valid_features.append((k, float_val))
        except:
            continue
    
    if valid_features:
        # 依值排序，取前 5 個最顯著的特徵
        valid_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, feature_value in valid_features[:5]:
            anomaly = analyze_feature_anomaly(feature_name, feature_value)
            if anomaly:
                report.append(f"- **{feature_name}**: `{feature_value:.2f}`")
                report.append(f"  {anomaly}")
            else:
                report.append(f"- **{feature_name}**: `{feature_value:.2f}`")
    else:
        report.append("無明顯異常特徵")
    
    report.append(f"")
    
    # 典型攻擊指標
    if attack_info['indicators']:
        report.append(f"### 📊 典型攻擊指標")
        for indicator in attack_info['indicators']:
            report.append(f"- {indicator}")
        report.append(f"")
    
    # 潛在風險
    if attack_info['risks']:
        report.append(f"### ⚠️ 潛在風險")
        for risk in attack_info['risks']:
            report.append(f"{risk}")
        report.append(f"")
    
    # 處理建議
    if attack_info['recommendations']:
        report.append(f"### 💡 處理建議")
        for i, rec in enumerate(attack_info['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append(f"")
    
    # 免責聲明
    report.append(f"---")
    report.append(f"*📌 此分析基於機器學習模型預測和領域知識規則，建議結合實際情況和專業資安團隊進行判斷。*")
    
    result = "\n".join(report)
    logger.info(f"威脅分析完成，報告長度: {len(result)} 字元")
    
    return result


if __name__ == "__main__":
    pass
