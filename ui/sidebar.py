import streamlit as st
import io
import requests
import joblib
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, clean_data
from src.feature_selector import run_genetic_selection
from src.model_trainer import train_and_evaluate

def display_sidebar():
    """
    Displays the sidebar UI components for model loading and training.
    """
    with st.sidebar:
        st.header("⚙️ 模型管理與訓練")

        # --- 從 URL 或本機檔案載入模型 ---
        with st.expander("載入預訓練模型", expanded=False):
            st.subheader("選項一：從 URL 載入")
            model_url = st.text_input("請輸入模型檔案的 Raw URL", help="請確保提供的是指向模型檔案本身的 Raw 連結。")
            if st.button("從 URL 載入模型"):
                if model_url:
                    with st.spinner("正在從 URL 下載並載入模型..."):
                        try:
                            response = requests.get(model_url)
                            response.raise_for_status()
                            
                            model_file = io.BytesIO(response.content)
                            loaded_data = joblib.load(model_file)
                            
                            st.session_state['trained_model'] = loaded_data['model']
                            st.session_state['scaler'] = loaded_data['scaler']
                            st.session_state['le'] = loaded_data['le']
                            st.session_state['selected_features'] = loaded_data['selected_features']
                            
                            st.session_state['model_loaded'] = True
                            st.session_state['selection_done'] = True
                            
                            with st.spinner("建立 SHAP 解釋器..."):
                                explainer = shap.TreeExplainer(st.session_state['trained_model'])
                                st.session_state['shap_explainer'] = explainer

                            st.success("模型從 URL 載入成功！")
                        except Exception as e:
                            st.error(f"從 URL 載入模型失敗：{e}")
                else:
                    st.warning("請先輸入模型檔案的 URL。")

            st.write("---")

            st.subheader("選項二：從本機檔案載入")
            uploaded_model_file = st.file_uploader("上傳 .joblib 模型檔案", type=['joblib'])
            if uploaded_model_file is not None:
                with st.spinner("正在從本機檔案載入模型..."):
                    try:
                        loaded_data = joblib.load(uploaded_model_file)
                        
                        st.session_state['trained_model'] = loaded_data['model']
                        st.session_state['scaler'] = loaded_data['scaler']
                        st.session_state['le'] = loaded_data['le']
                        st.session_state['selected_features'] = loaded_data['selected_features']
                        
                        st.session_state['model_loaded'] = True
                        st.session_state['selection_done'] = True
                        
                        with st.spinner("建立 SHAP 解釋器..."):
                            explainer = shap.TreeExplainer(st.session_state['trained_model'])
                            st.session_state['shap_explainer'] = explainer

                        st.success("模型從本機檔案載入成功！")
                    except Exception as e:
                        st.error(f"從本機檔案載入模型失敗：{e}")

        st.write("---")

        # ==============================================================================
        # 流程一：本機訓練流程
        # ==============================================================================
        # This part runs only if no model has been loaded yet.
        if not st.session_state.get('model_loaded', False):
            st.header("本機訓練流程")
            st.info("偵測到無預載模型，您可以在此執行完整的資料讀取與訓練流程。")
            DATA_PATH = "data/03-01-2018.csv"
            
            # Use session state to cache the loaded and cleaned dataframe
            if 'df_cleaned' not in st.session_state:
                if st.button("1. 載入與清理資料"):
                    with st.spinner("載入原始資料..."):
                        df_raw = load_data(DATA_PATH)
                        if df_raw is not None:
                            feature_cols = df_raw.columns.drop(['Label', 'Timestamp'])
                            for col in feature_cols:
                                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                            st.session_state['df_cleaned'] = clean_data(df_raw.copy())
                            st.success(f"資料載入與清理完成！")
                            st.rerun()
                        else:
                            st.error(f"無法從 {DATA_PATH} 載入資料。")
            
            if 'df_cleaned' in st.session_state:
                st.success("步驟 1：資料已載入")

                # --- 特徵選擇 ---
                if st.button("2. 開始特徵選擇"):
                    df_cleaned = st.session_state['df_cleaned']
                    with st.spinner("正在進行資料預處理..."):
                        X = df_cleaned.drop(columns=['Label', 'Timestamp'])
                        y = df_cleaned['Label']
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
                        st.session_state['scaler'] = scaler
                    st.success("資料預處理完成！")

                    with st.spinner("執行基因演算法中..."):
                        selected_features, best_score = run_genetic_selection(X_scaled, y_encoded)
                    
                    st.session_state['best_ga_score'] = best_score
                    st.session_state['num_total_features'] = len(X.columns)
                    st.session_state['selection_done'] = True
                    st.session_state['selected_features'] = selected_features
                    st.session_state['X_scaled'] = X_scaled
                    st.session_state['y_encoded'] = y_encoded
                    st.session_state['le'] = le
                    st.success("步驟 2：特徵選擇完成！結果請至儀表板查看。")
                    st.rerun()

                if st.session_state.get('selection_done', False):
                    st.success("步驟 2：特徵選擇已完成")
                    # --- 模型訓練 ---
                    if st.button("3. 訓練模型"):
                        with st.spinner("正在準備訓練資料..."):
                            X_selected = st.session_state['X_scaled'][st.session_state['selected_features']]
                            y_encoded = st.session_state['y_encoded']
                            le = st.session_state['le']
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                            )
                        st.success("資料分割完成！")

                        with st.spinner("模型訓練與評估中..."):
                            metrics, model, cm_df = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_)
                        
                        st.session_state['trained_model'] = model
                        st.session_state['metrics'] = metrics
                        st.session_state['cm_df'] = cm_df

                        with st.spinner("建立 SHAP 解釋器..."):
                            explainer = shap.TreeExplainer(model)
                            st.session_state['shap_explainer'] = explainer
                        st.success("步驟 3：模型訓練完成！評估結果請至儀表板查看。")
                        st.rerun()
            
            if st.session_state.get('trained_model'):
                st.success("步驟 3：模型已訓練")
                # --- 儲存模型區塊 ---
                st.subheader("儲存已訓練模型")
                st.info("將目前訓練好的模型、特徵列表與所有相關設定打包儲存。")
                if st.button("💾 儲存模型"):
                    with st.spinner("正在打包並儲存模型..."):
                        try:
                            data_to_save = {
                                'model': st.session_state['trained_model'],
                                'scaler': st.session_state['scaler'],
                                'le': st.session_state['le'],
                                'selected_features': st.session_state['selected_features']
                            }
                            filename = "ids_model_package.joblib"
                            joblib.dump(data_to_save, filename)
                            st.success(f"模型已成功儲存為 **{filename}**！")
                        except Exception as e:
                            st.error(f"儲存模型時發生錯誤：{e}")