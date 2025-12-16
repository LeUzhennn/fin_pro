import streamlit as st
from typing import Any

# Import UI components from the ui directory
from ui.sidebar import display_sidebar
from ui.tab_dashboard import display_dashboard_tab
from ui.tab_single_prediction import display_single_prediction_tab
from ui.tab_batch_prediction import display_batch_prediction_tab

# Import configuration
from config import UIConfig
from src.logger import get_logger

logger = get_logger(__name__)

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    # ==============================================================================
    # Main App Configuration
    # ==============================================================================

    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout=UIConfig.LAYOUT
    )
    
    logger.info("應用程式啟動")

    st.title(f"{UIConfig.PAGE_ICON} {UIConfig.PAGE_TITLE}")

    # ==============================================================================
    # Sidebar
    # ==============================================================================
    try:
        display_sidebar()
    except Exception as e:
        st.sidebar.error(f"側邊欄載入失敗: {e}")

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
        try:
            display_dashboard_tab()
        except Exception as e:
            st.error(f"儀表板載入失敗: {e}")

    with tab2:
        if st.session_state.get('trained_model'):
            try:
                display_single_prediction_tab()
            except Exception as e:
                st.error(f"單筆預測功能載入失敗: {e}")
        else:
            st.info("請先從側邊欄載入或訓練一個模型，才能使用即時預測功能。")

    with tab3:
        if st.session_state.get('trained_model'):
            try:
                display_batch_prediction_tab()
            except Exception as e:
                st.error(f"批次分析功能載入失敗: {e}")
        else:
            st.info("請先從側邊欄載入或訓練一個模型，才能使用批次分析功能。")

if __name__ == "__main__":
    main()