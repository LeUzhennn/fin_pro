import streamlit as st

# Import UI components from the ui directory
from ui.sidebar import display_sidebar
from ui.tab_dashboard import display_dashboard_tab
from ui.tab_single_prediction import display_single_prediction_tab
from ui.tab_batch_prediction import display_batch_prediction_tab

# ==============================================================================
# Main App Configuration
# ==============================================================================

st.set_page_config(
    page_title="入侵偵測互動式分析系統",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 入侵偵測互動式分析系統")

# ==============================================================================
# Sidebar
# ==============================================================================
display_sidebar()


# ==============================================================================
# Main Content Area with Tabs
# ==============================================================================

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📈 儀表板 & 模型評估", 
    "🔬 即時單筆預測", 
    "🗂️ 批次流量分析"
])

# Populate each tab
with tab1:
    display_dashboard_tab()

with tab2:
    if st.session_state.get('trained_model'):
        display_single_prediction_tab()
    else:
        st.info("請先從側邊欄載入或訓練一個模型，才能使用即時預測功能。" )

with tab3:
    if st.session_state.get('trained_model'):
        display_batch_prediction_tab()
    else:
        st.info("請先從側邊欄載入或訓練一個模型，才能使用批次分析功能。" )
