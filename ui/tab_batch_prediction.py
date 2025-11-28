import streamlit as st
import pandas as pd
import numpy as np

# We need the summary function
from ui.utils import generate_shap_summary

def display_batch_prediction_tab():
    """
    Displays the UI for the Batch Analysis tab.
    """
    st.header("🗂️ 批次流量分析")

    if 'selected_features' not in st.session_state:
        st.warning("模型尚未載入或訓練，無法進行批次分析。")
        return

    st.write("上傳包含多筆網路流量的 CSV 檔，系統將逐筆分析並判斷是否為攻擊。")

    # --- Template Download ---
    template_df = pd.DataFrame(columns=st.session_state['selected_features'])
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下載分析範例 CSV 檔案",
        data=csv_template,
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    # --- File Uploader ---
    uploaded_file = st.file_uploader("上傳待分析的 CSV 檔案", type=["csv"])

    if uploaded_file is not None:
        # Clear previous results if a new file is uploaded
        if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            st.session_state.current_file_name = uploaded_file.name
            if 'batch_results_df' in st.session_state:
                del st.session_state['batch_results_df']

        try:
            batch_df_raw = pd.read_csv(uploaded_file)
            batch_df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            with st.expander("點此查看上傳的原始資料 (前 5 筆)"):
                st.dataframe(batch_df_raw.head())

            # --- Column Mapping ---
            uploaded_columns = batch_df_raw.columns.tolist()
            st.subheader("欄位映射設定")
            column_mapping = {}
            mapping_cols = st.columns(4)
            for i, feature in enumerate(st.session_state['selected_features']):
                with mapping_cols[i % 4]:
                    default_index = uploaded_columns.index(feature) + 1 if feature in uploaded_columns else 0
                    column_mapping[feature] = st.selectbox(
                        f"模型特徵: {feature}",
                        ['未映射'] + uploaded_columns,
                        index=default_index,
                        key=f"map_{feature}"
                    )
            
            # --- Run Analysis ---
            if st.button("🚀 開始分析流量"):
                with st.spinner("正在根據映射設定處理資料並進行分析..."):
                    # Create a DataFrame with the correct feature columns
                    batch_X_mapped_user = pd.DataFrame(0.0, index=batch_df_raw.index, columns=st.session_state['selected_features'])
                    for model_feature, uploaded_col in column_mapping.items():
                        if uploaded_col != '未映射':
                            batch_X_mapped_user[model_feature] = pd.to_numeric(batch_df_raw[uploaded_col], errors='coerce')
                    
                    batch_X_mapped_user.replace([np.inf, -np.inf], np.nan, inplace=True)
                    batch_X_mapped_user.dropna(inplace=True)

                    if batch_X_mapped_user.empty:
                        st.warning("預處理後，上傳檔案中沒有有效資料可供分析。")
                        if 'batch_results_df' in st.session_state:
                            del st.session_state['batch_results_df']
                    else:
                        scaler = st.session_state['scaler']
                        model = st.session_state['trained_model']
                        le = st.session_state['le']

                        required_features_for_scaler = scaler.feature_names_in_
                        batch_df_full = pd.DataFrame(0.0, index=batch_X_mapped_user.index, columns=required_features_for_scaler)
                        
                        for col in batch_X_mapped_user.columns:
                            if col in batch_df_full.columns:
                                batch_df_full[col] = batch_X_mapped_user[col]

                        batch_scaled_full = scaler.transform(batch_df_full)
                        batch_scaled_df = pd.DataFrame(batch_scaled_full, index=batch_df_full.index, columns=required_features_for_scaler)
                        final_batch_for_model = batch_scaled_df[st.session_state['selected_features']]
                        st.session_state['final_batch_for_model'] = final_batch_for_model

                        batch_predictions_encoded = model.predict(final_batch_for_model)
                        batch_predictions_label = le.inverse_transform(batch_predictions_encoded)

                        batch_df_results = batch_df_raw.loc[final_batch_for_model.index].copy()
                        batch_df_results['Predicted_Label'] = batch_predictions_label
                        batch_df_results['分析結果'] = batch_df_results['Predicted_Label'].apply(lambda x: '攻擊' if x != 'Benign' else '正常')
                        st.session_state['batch_results_df'] = batch_df_results

        except Exception as e:
            st.error(f"處理上傳檔案時發生錯誤：{e}")
            if 'batch_results_df' in st.session_state:
                del st.session_state['batch_results_df']

        # --- Display Results ---
        if 'batch_results_df' in st.session_state:
            batch_df_results = st.session_state['batch_results_df']
            
            st.subheader("📊 分析結果總覽")
            prediction_counts = batch_df_results['分析結果'].value_counts()
            st.bar_chart(prediction_counts)

            st.subheader("📄 詳細分析結果")
            filter_option = st.radio(
                "篩選顯示結果：",
                ('顯示全部', '僅顯示攻擊', '僅顯示正常'),
                horizontal=True,
                key='filter_radio'
            )

            if filter_option == '僅顯示攻擊':
                filtered_df = batch_df_results[batch_df_results['分析結果'] == '攻擊']
            elif filter_option == '僅顯示正常':
                filtered_df = batch_df_results[batch_df_results['分析結果'] == '正常']
            else:
                filtered_df = batch_df_results

            if not filtered_df.empty:
                st.dataframe(filtered_df)
                csv_results = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下載目前的分析結果",
                    data=csv_results,
                    file_name="traffic_analysis_results.csv",
                    mime="text/csv"
                )

            # --- SHAP Drill-down ---
            st.subheader("🔬 深入分析單筆攻擊流量 (SHAP Drill-down)")
            attack_df = batch_df_results[batch_df_results['分析結果'] == '攻擊']
            if attack_df.empty:
                st.info("在目前的分析結果中，沒有偵測到攻擊流量可供深入分析。")
            else:
                selected_index = st.selectbox(
                    "選擇一筆攻擊流量的索引 (Index) 進行分析：",
                    options=attack_df.index
                )

                if selected_index is not None:
                    with st.spinner("正在為您選擇的流量產生 SHAP 分析..."):
                        try:
                            explainer = st.session_state['shap_explainer']
                            final_batch_for_model = st.session_state['final_batch_for_model']
                            le = st.session_state['le']

                            single_instance = final_batch_for_model.loc[[selected_index]]
                            single_prediction_label = batch_df_results.loc[selected_index, 'Predicted_Label']
                            single_prediction_index = list(le.classes_).index(single_prediction_label)

                            shap_values = explainer.shap_values(single_instance)
                            
                            base_value = explainer.expected_value
                            if isinstance(base_value, (list, np.ndarray)):
                                if single_prediction_index < len(base_value):
                                    shap_base_value = base_value[single_prediction_index]
                                else:
                                    shap_base_value = base_value[0]
                            else:
                                shap_base_value = base_value

                            if isinstance(shap_values, list):
                                if single_prediction_index < len(shap_values):
                                    shap_values_for_class = shap_values[single_prediction_index]
                                else:
                                    shap_values_for_class = shap_values[0]
                            else:
                                if single_prediction_index == 0:
                                    shap_values_for_class = -shap_values
                                else:
                                    shap_values_for_class = shap_values
                                    
                            shap_values_for_class = shap_values_for_class.flatten()

                            features_for_plot = single_instance.iloc[0]
                            if len(shap_values_for_class) == len(features_for_plot) + 1:
                                shap_values_for_class = shap_values_for_class[:-1]

                            summary_text = generate_shap_summary(
                                shap_values_for_class, 
                                single_instance,
                                single_prediction_label,
                                le,
                                shap_base_value
                            )
                            st.markdown(summary_text)
                        except KeyError:
                            st.error(f"發生錯誤：無法在已處理的資料中找到索引 {selected_index}。")
                        except Exception as e:
                            st.warning(f"無法產生 SHAP 分析：{e}")