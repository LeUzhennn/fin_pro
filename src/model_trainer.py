"""此模組負責模型訓練、評估與預測。"""
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple, Dict
import numpy as np

from config import ModelConfig
from src.logger import get_logger

logger = get_logger(__name__)

@st.cache_data(show_spinner=False)
def train_and_evaluate(_X_train, _X_test, _y_train, _y_test, class_names) -> Tuple[Dict[str, float], RandomForestClassifier, pd.DataFrame]:
    """
    訓練隨機森林模型並評估其效能。
    
    Args:
        _X_train: 訓練特徵
        _X_test: 測試特徵
        _y_train: 訓練標籤
        _y_test: 測試標籤
        class_names: 類別名稱列表
        
    Returns:
        Tuple[Dict, RandomForestClassifier, pd.DataFrame]: 評估指標、訓練好的模型、混淆矩陣
    """
    logger.info(f"開始訓練隨機森林模型: n_estimators={ModelConfig.RF_N_ESTIMATORS}")
    with st.spinner("正在訓練隨機森林模型..."):
        model = RandomForestClassifier(
            n_estimators=ModelConfig.RF_N_ESTIMATORS,
            random_state=ModelConfig.RF_RANDOM_STATE,
            n_jobs=ModelConfig.RF_N_JOBS
        )
        model.fit(_X_train, _y_train)
        logger.info("模型訓練完成")

    with st.spinner("正在評估模型效能..."):
        y_pred = model.predict(_X_test)

        metrics = {
            "accuracy": accuracy_score(_y_test, y_pred),
            "precision": precision_score(_y_test, y_pred, average='weighted'),
            "recall": recall_score(_y_test, y_pred, average='weighted'),
            "f1_score": f1_score(_y_test, y_pred, average='weighted')
        }

        # 計算混淆矩陣
        cm = confusion_matrix(_y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return metrics, model, cm_df