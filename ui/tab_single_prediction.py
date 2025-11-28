import streamlit as st
import pandas as pd
import numpy as np

# We need the summary function
from ui.utils import generate_shap_summary

def display_single_prediction_tab():
    """
    Displays the UI for the Live Prediction tab.
    """
    st.header("🔬 即時單筆預測")
    st.write("請輸入以下特徵值，來模擬一筆新的網路流量數據：")

    if 'selected_features' not in st.session_state:
        st.warning("模型尚未載入或訓練，無法進行預測。")
        return

    selected_features = st.session_state['selected_features']
    
    with st.form(key='prediction_form'):
        num_cols = 4  # You can adjust the number of columns
        cols = st.columns(num_cols)
        user_inputs = {}
        for i, feature in enumerate(selected_features):
            with cols[i % num_cols]:
                user_inputs[feature] = st.number_input(label=feature, value=0.0, format="%.4f", key=f"input_{feature}")
        
        submit_button = st.form_submit_button(label='⚡ 執行預測')

    if submit_button:
        with st.spinner("執行預測與分析中..."):
            input_df_user = pd.DataFrame([user_inputs])
            scaler = st.session_state['scaler']
            model = st.session_state['trained_model']
            le = st.session_state['le']
            
            # Ensure the input DataFrame has all columns the scaler expects
            required_features_for_scaler = scaler.feature_names_in_
            input_df_full = pd.DataFrame(0.0, index=[0], columns=required_features_for_scaler)
            for col in input_df_user.columns:
                if col in input_df_full.columns:
                    input_df_full[col] = input_df_user[col].values

            # Scaling and prediction
            input_scaled_full = scaler.transform(input_df_full)
            input_scaled_df = pd.DataFrame(input_scaled_full, columns=required_features_for_scaler)
            final_input_for_model = input_scaled_df[st.session_state['selected_features']]

            prediction = model.predict(final_input_for_model)
            predicted_label = le.inverse_transform(prediction)[0]

            st.subheader("預測結果")
            if predicted_label == 'Benign':
                st.success(f"✅ 預測結果： **{predicted_label}** (正常)")
            else:
                st.error(f"🚨 預測結果： **{predicted_label}** (攻擊!)")

            # SHAP Analysis
            st.subheader("模型預測解釋 (SHAP Analysis)")
            try:
                explainer = st.session_state['shap_explainer']
                shap_values = explainer.shap_values(final_input_for_model)
                predicted_class_index = prediction[0]

                # --- START of the new, safe logic ---
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if predicted_class_index < len(base_value):
                        shap_base_value = base_value[predicted_class_index]
                    else:
                        shap_base_value = base_value[0]
                else:
                    shap_base_value = base_value

                if isinstance(shap_values, list):
                    if predicted_class_index < len(shap_values):
                        shap_values_for_class = shap_values[predicted_class_index]
                    else:
                        shap_values_for_class = shap_values[0]
                else:
                    if predicted_class_index == 0:
                        shap_values_for_class = -shap_values
                    else:
                        shap_values_for_class = shap_values
                        
                shap_values_for_class = shap_values_for_class.flatten()

                features_for_plot = final_input_for_model.iloc[0]
                if len(shap_values_for_class) == len(features_for_plot) + 1:
                    shap_values_for_class = shap_values_for_class[:-1]
                # --- END of the new, safe logic ---

                summary_text = generate_shap_summary(
                    shap_values_for_class, 
                    final_input_for_model, 
                    predicted_label, 
                    le,
                    shap_base_value
                )
                st.markdown(summary_text)
            except Exception as e:
                st.warning(f"無法產生 SHAP 分析：{e}")