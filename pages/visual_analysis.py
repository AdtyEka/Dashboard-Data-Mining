"""Halaman Analisis Visual"""
import streamlit as st
import plotly.express as px
from utils.visualizations import create_histogram, create_scatter, create_box_plot


def render_visual_analysis(filtered_df):
    """Render halaman analisis visual"""
    st.title("Analisis Visual Data Stunting")
    st.markdown("---")
    
    viz_options = []
    if 'Age' in filtered_df.columns and 'Stunting' in filtered_df.columns:
        viz_options.append("Distribusi Umur")
    if 'Body_Weight' in filtered_df.columns and 'Body_Length' in filtered_df.columns:
        viz_options.append("Hubungan Berat & Panjang Badan")
    if 'Birth_Weight' in filtered_df.columns and 'Birth_Length' in filtered_df.columns:
        viz_options.append("Hubungan Berat & Panjang Lahir")
    if 'Body_Weight' in filtered_df.columns:
        viz_options.append("Distribusi Berat Badan")
    if 'Body_Length' in filtered_df.columns:
        viz_options.append("Distribusi Panjang Badan")
    if len([c for c in ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting'] if c in filtered_df.columns]) >= 3:
        viz_options.append("Heatmap Korelasi")
    
    if not viz_options:
        st.warning("Dataset yang dipilih tidak memiliki kolom yang cukup untuk visualisasi.")
        return
    
    viz_type = st.selectbox("Pilih Jenis Visualisasi", viz_options)
    
    if viz_type == "Distribusi Umur":
        st.subheader("Distribusi Umur berdasarkan Status Stunting")
        fig = create_histogram(filtered_df, 'Age', 'Stunting', "Distribusi Umur")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Badan":
        st.subheader("Hubungan Berat Badan vs Panjang Badan")
        fig = create_scatter(
            filtered_df,
            'Body_Length',
            'Body_Weight',
            'Stunting',
            'Age' if 'Age' in filtered_df.columns else None,
            "Hubungan Berat Badan vs Panjang Badan"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Lahir":
        st.subheader("Hubungan Berat Lahir vs Panjang Lahir")
        fig = create_scatter(
            filtered_df,
            'Birth_Length',
            'Birth_Weight',
            'Stunting',
            'Age' if 'Age' in filtered_df.columns else None,
            "Hubungan Berat Lahir vs Panjang Lahir"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribusi Berat Badan":
        st.subheader("Distribusi Berat Badan")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_box_plot(filtered_df, 'Stunting', 'Body_Weight', 'Stunting', "Box Plot Berat Badan")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_histogram(filtered_df, 'Body_Weight', 'Stunting', "Histogram Berat Badan")
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Distribusi Panjang Badan":
        st.subheader("Distribusi Panjang Badan")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_box_plot(filtered_df, 'Stunting', 'Body_Length', 'Stunting', "Box Plot Panjang Badan")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_histogram(filtered_df, 'Body_Length', 'Stunting', "Histogram Panjang Badan")
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Heatmap Korelasi":
        st.subheader("Heatmap Korelasi Variabel Numerik")
        numeric_cols = [c for c in ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting'] if c in filtered_df.columns]
        if len(numeric_cols) >= 2:
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                labels=dict(x="Variabel", y="Variabel", color="Korelasi")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

