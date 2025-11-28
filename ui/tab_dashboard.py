import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from src.data_loader import load_data, clean_data

def display_dashboard_tab():
    """
    Displays the UI for the Dashboard & Model Evaluation tab.
    """
    st.header("📈 儀表板 & 模型評估")

    # This tab shows different things depending on what's available in session_state
    
    # If a model is trained or loaded, show its performance
    if st.session_state.get('trained_model'):
        st.subheader("模型評估指標")
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        else:
            st.info("模型已載入，但無評估指標可顯示 (可能為外部載入的模型)。")

        st.subheader("混淆矩陣 (Confusion Matrix)")
        if 'cm_df' in st.session_state:
            cm_df = st.session_state['cm_df']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("模型已載入，但無混淆矩陣可顯示。")

    # If feature selection is done, show the results
    if st.session_state.get('selection_done', False):
        st.subheader("基因演算法選擇結果")
        if 'best_ga_score' in st.session_state:
            st.success(f"演算法執行完畢！最佳分數 (Accuracy): {st.session_state['best_ga_score']:.4f}")
            st.metric(label="選擇的特徵數量", value=f"{len(st.session_state['selected_features'])} / {st.session_state['num_total_features']}")
        
        st.write("**選擇的特徵列表：**")
        st.dataframe(st.session_state['selected_features'])
    
    st.write("---")

    # Show data analysis if data is loaded
    if 'df_cleaned' in st.session_state:
        df_cleaned = st.session_state['df_cleaned']
        st.header("資料集分析")
        st.subheader("**目標變數 (Label) 分析**")
        label_counts = df_cleaned['Label'].value_counts()
        st.write("各類別資料筆數：")
        st.write(label_counts)
        st.subheader("目標變數分佈圖")
        st.bar_chart(label_counts)
        st.info("從上圖可知，資料集存在嚴重的類別不平衡問題。")

        with st.expander("顯示清理後的資料摘要"):
            st.subheader("資料預覽 (前 5 筆)")
            st.write(df_cleaned.head())
            st.subheader("資料基本資訊")
            buffer = io.StringIO()
            df_cleaned.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.subheader("數值特徵統計摘要")
            st.write(df_cleaned.describe())
    else:
        st.info("請至側邊欄點擊「1. 載入與清理資料」以開始。")