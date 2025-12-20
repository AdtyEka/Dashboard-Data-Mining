"""Halaman Data Explorer"""
import streamlit as st


def render_data_explorer(filtered_df):
    """Render halaman data explorer"""
    st.title("Data Explorer")
    st.markdown("---")
    
    st.subheader("Tabel Data")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data sebagai CSV",
        data=csv,
        file_name='filtered_stunting_data_combined.csv',
        mime='text/csv'
    )
    
    st.markdown("---")
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kolom Dataset:**")
        st.write(list(filtered_df.columns))
    
    with col2:
        st.write("**Tipe Data:**")
        st.write(filtered_df.dtypes)
    
    st.write("**Shape Dataset:**", filtered_df.shape)
    st.write("**Missing Values:**")
    st.write(filtered_df.isnull().sum())

