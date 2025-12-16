import streamlit as st
import pandas as pd
import numpy as np

# Import the new LLM analyzer function
from src.llm_analyzer import get_threat_explanation

# We need the summary function
from ui.utils import generate_shap_summary

@st.cache_data
def get_samples():
    """Loads one 'Benign' and one 'Infilteration' sample from the dataset."""
    try:
        # Read the full dataset to ensure we find the samples.
        # This is cached, so it only runs on the first load.
        df = pd.read_csv("data/03-01-2018.csv", low_memory=False)
        df.columns = df.columns.str.strip()
        df['Label'] = df['Label'].str.strip()

        # Clean numeric-like columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['Label', 'Timestamp']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        
        samples_dict = {}

        # Safely get a benign sample
        benign_df = df[df['Label'] == 'Benign']
        if not benign_df.empty:
            samples_dict["範例 1：正常的 (Benign) 流量"] = benign_df.iloc[0].to_dict()
        else:
            st.warning("數據集中找不到 'Benign' 範例。")

        # Safely get an infilteration sample
        infilteration_df = df[df['Label'] == 'Infilteration']
        if not infilteration_df.empty:
            samples_dict["範例 2：入侵的 (Infilteration) 流量"] = infilteration_df.iloc[0].to_dict()
        else:
            st.warning("數據集中找不到 'Infilteration' 範例。")

        return samples_dict
    except Exception as e:
        st.error(f"無法載入範例數據: {e}")
        return {}

def display_single_prediction_tab():
    """
    Displays the UI for the Live Prediction tab, now with robust state management.
    """
    st.header("🔬 即時單筆預測")
    
    if 'selected_features' not in st.session_state:
        st.warning("模型尚未載入或訓練，無法進行預測。")
        return

    selected_features = st.session_state['selected_features']
    samples = get_samples()

    # --- Initialize State ---
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "last_user_inputs" not in st.session_state:
        st.session_state.last_user_inputs = {}
    if "llm_explanation" not in st.session_state:
        st.session_state.llm_explanation = None

    # --- Sample Data Loader Logic ---
    def update_form_with_sample():
        sample_key = st.session_state.get("sample_selector")
        sample_data = samples.get(sample_key, {})
        for feature in selected_features:
            st.session_state[f"input_{feature}"] = float(sample_data.get(feature, 0.0))
        # Clear previous results when a new sample is loaded
        st.session_state.prediction_result = None
        st.session_state.llm_explanation = None

    st.markdown("#### 載入範例數據以自動填單")
    sample_options = ["-- 請選擇一筆範例資料 --"] + list(samples.keys())
    st.selectbox("選擇一筆範例", options=sample_options, key="sample_selector", on_change=update_form_with_sample)
    
    st.write("---")
    st.write("請輸入或修改以下特徵值，來模擬一筆新的網路流量數據：")

    # --- Prediction Form ---
    with st.form(key='prediction_form'):
        num_cols = 4
        cols = st.columns(num_cols)
        current_inputs = {}
        for i, feature in enumerate(selected_features):
            with cols[i % num_cols]:
                current_inputs[feature] = st.number_input(label=feature, format="%.4f", key=f"input_{feature}")
        submit_button = st.form_submit_button(label='⚡ 執行預測')

    # --- Processing and State Update (on form submission) ---
    if submit_button:
        st.session_state.last_user_inputs = current_inputs
        st.session_state.llm_explanation = None # Clear old explanation
        
        with st.spinner("執行預測中..."):
            input_df_user = pd.DataFrame([current_inputs])
            scaler = st.session_state['scaler']
            model = st.session_state['trained_model']
            le = st.session_state['le']
            
            required_features_for_scaler = scaler.feature_names_in_
            input_df_full = pd.DataFrame(0.0, index=[0], columns=required_features_for_scaler)
            for col in input_df_user.columns:
                if col in input_df_full.columns:
                    input_df_full[col] = input_df_user[col].values

            input_scaled_full = scaler.transform(input_df_full)
            input_scaled_df = pd.DataFrame(input_scaled_full, columns=required_features_for_scaler)
            final_input_for_model = input_scaled_df[st.session_state['selected_features']]

            prediction = model.predict(final_input_for_model)
            predicted_label = le.inverse_transform(prediction)[0]
            st.session_state.prediction_result = predicted_label
            # The rest of the display logic is now outside this block

    # --- Display Area (depends on session_state, not form submission) ---
    if st.session_state.prediction_result:
        predicted_label = st.session_state.prediction_result
        user_inputs = st.session_state.last_user_inputs

        st.subheader("預測結果")
        if predicted_label == 'Benign':
            st.success(f"✅ 預測結果： **{predicted_label}** (正常)")
        else:
            st.error(f"🚨 預測結果： **{predicted_label}** (攻擊!)")

        # --- LLM Analysis Section ---
        if predicted_label != 'Benign':
            if st.button("🤖 AI 分析原因"):
                with st.spinner("正在呼叫資深AI分析師，請稍候..."):
                    explanation = get_threat_explanation(user_inputs, predicted_label)
                    st.session_state.llm_explanation = explanation
        
        if st.session_state.llm_explanation:
            st.subheader("AI 威脅分析報告")
            st.markdown(st.session_state.llm_explanation)
        # --- End LLM Analysis Section ---

        # --- SHAP Analysis Section ---
        st.subheader("模型預測解釋 (SHAP Analysis)")
        try:
            with st.spinner("正在計算 SHAP 值..."):
                # Recalculate necessary values for SHAP display
                input_df_user = pd.DataFrame([user_inputs])
                scaler = st.session_state['scaler']
                le = st.session_state['le']
                required_features_for_scaler = scaler.feature_names_in_
                input_df_full = pd.DataFrame(0.0, index=[0], columns=required_features_for_scaler)
                for col in input_df_user.columns:
                    if col in input_df_full.columns:
                        input_df_full[col] = input_df_user[col].values
                input_scaled_full = scaler.transform(input_df_full)
                input_scaled_df = pd.DataFrame(input_scaled_full, columns=required_features_for_scaler)
                final_input_for_model = input_scaled_df[st.session_state['selected_features']]
                prediction = st.session_state['trained_model'].predict(final_input_for_model)
                predicted_class_index = prediction[0]

                explainer = st.session_state['shap_explainer']
                shap_values = explainer.shap_values(final_input_for_model)

                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    shap_base_value = base_value[predicted_class_index] if predicted_class_index < len(base_value) else base_value[0]
                else:
                    shap_base_value = base_value

                if isinstance(shap_values, list):
                    shap_values_for_class = shap_values[predicted_class_index] if predicted_class_index < len(shap_values) else shap_values[0]
                else:
                    shap_values_for_class = -shap_values if predicted_class_index == 0 else shap_values
                
                shap_values_for_class = shap_values_for_class.flatten()
                
                summary_text = generate_shap_summary(shap_values_for_class, final_input_for_model, predicted_label, le, shap_base_value)
                st.markdown(summary_text)
        except Exception as e:
            st.warning(f"無法產生 SHAP 分析：{e}")