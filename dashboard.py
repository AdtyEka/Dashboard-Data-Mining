"""Dashboard Analisis Stunting - File Utama"""
import streamlit as st
from utils.data_loader import load_data
from utils.filters import setup_sidebar_filters
from modules.overview import render_overview
from modules.visual_analysis import render_visual_analysis
from modules.detail_analysis import render_detail_analysis
from modules.data_explorer import render_data_explorer
from modules.prediction import render_prediction

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Stunting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data (gabungkan semua dataset)
df = load_data()

# Setup sidebar dan filter
page, filtered_df = setup_sidebar_filters(df)

# Routing halaman
if page == "Overview":
    render_overview(filtered_df)
elif page == "Analisis Visual":
    render_visual_analysis(filtered_df)
elif page == "Analisis Detail":
    render_detail_analysis(filtered_df)
elif page == "Data Explorer":
    render_data_explorer(filtered_df)
elif page == "Prediksi":
    render_prediction()
