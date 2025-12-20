"""Fungsi untuk filter sidebar"""
import streamlit as st
import pandas as pd


def count_stunting(stunting_series):
    """Hitung jumlah kasus stunting, handle baik numerik maupun string"""
    if stunting_series.empty:
        return 0
    
    # Cek apakah kolom numerik
    if pd.api.types.is_numeric_dtype(stunting_series):
        # Jika numerik, hitung yang == 1 atau > 0
        return int((stunting_series == 1).sum() if (stunting_series == 1).any() else (stunting_series > 0).sum())
    else:
        # Jika string, hitung yang bernilai positif (case-insensitive)
        stunting_series_lower = stunting_series.astype(str).str.lower().str.strip()
        positive_values = ['yes', 'stunting', '1', 'true', 'y']
        return int(stunting_series_lower.isin(positive_values).sum())


def setup_sidebar_filters(df):
    """Setup filter di sidebar dan return filtered dataframe"""
    st.sidebar.title("Navigasi Dashboard")
    st.sidebar.markdown("---")
    
    # Menu navigasi
    page = st.sidebar.radio(
        "Pilih Halaman",
        ["Overview", "Analisis Visual", "Analisis Detail", "Data Explorer", "Prediksi"]
    )
    
    # Filter di sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Data")
    
    # Filter berdasarkan jenis kelamin
    if 'Sex' in df.columns:
        sex_filter = st.sidebar.multiselect(
            "Jenis Kelamin",
            options=df['Sex'].unique(),
            default=df['Sex'].unique()
        )
    else:
        sex_filter = []
    
    # Filter berdasarkan ASI Eksklusif
    if 'ASI_Eksklusif' in df.columns:
        asi_filter = st.sidebar.multiselect(
            "ASI Eksklusif",
            options=df['ASI_Eksklusif'].unique(),
            default=df['ASI_Eksklusif'].unique()
        )
    else:
        asi_filter = []
    
    # Filter berdasarkan Stunting
    if 'Stunting' in df.columns:
        stunting_filter = st.sidebar.multiselect(
            "Status Stunting",
            options=df['Stunting'].unique(),
            default=df['Stunting'].unique()
        )
    else:
        stunting_filter = []
    
    # Filter umur
    if 'Age' in df.columns:
        age_range = st.sidebar.slider(
            "Rentang Umur (bulan)",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max()))
        )
    else:
        age_range = (0, 100)
    
    # Terapkan filter
    filtered_df = df.copy()
    if 'Sex' in df.columns:
        filtered_df = filtered_df[filtered_df['Sex'].isin(sex_filter)]
    if 'ASI_Eksklusif' in df.columns:
        filtered_df = filtered_df[filtered_df['ASI_Eksklusif'].isin(asi_filter)]
    if 'Stunting' in df.columns:
        filtered_df = filtered_df[filtered_df['Stunting'].isin(stunting_filter)]
    if 'Age' in df.columns:
        filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    
    # Informasi di sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Data", f"{len(filtered_df):,}")
    if 'Stunting' in filtered_df.columns:
        # Hitung jumlah stunting (handle baik numerik maupun string)
        stunting_count = count_stunting(filtered_df['Stunting'])
        st.sidebar.metric("Data Stunting", f"{stunting_count:,}")
        if len(filtered_df) > 0:
            st.sidebar.metric("Persentase Stunting", f"{(stunting_count/len(filtered_df)*100):.2f}%")
        else:
            st.sidebar.metric("Persentase Stunting", "0.00%")
    
    # Informasi dataset yang digabung
    if 'Dataset_Source' in df.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset yang Digabung")
        dataset_counts = df['Dataset_Source'].value_counts()
        for source, count in dataset_counts.items():
            st.sidebar.text(f"{source}: {count:,} data")
    
    return page, filtered_df

