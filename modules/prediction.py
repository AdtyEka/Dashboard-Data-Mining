"""Halaman Prediksi"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.model_utils import MODEL_AVAILABLE, load_model, preprocess_input, interpret_prediction
from constants import MODEL_PATH, COLORS


def render_prediction():
    """Render halaman prediksi"""
    st.title("Prediksi Stunting")
    st.markdown("---")
    
    if not MODEL_AVAILABLE:
        st.error("TensorFlow/Keras tidak terinstall. Install dengan: `pip install tensorflow`")
        st.code("pip install tensorflow", language="bash")
        return
    
    # Tampilkan loading indicator
    with st.spinner("Memuat model..."):
        model = load_model(MODEL_PATH)
    
    if model is None:
        # Error sudah ditampilkan oleh load_model
        st.info("**Tips:** Pastikan file model `best_stunting_model.h5` ada di folder yang sama dengan `dashboard.py`")
        return
    
    with st.expander("Informasi Model"):
        try:
            st.write(f"**Nama Model:** {MODEL_PATH}")
            st.write(f"**Jumlah Layer:** {len(model.layers)}")
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
            st.write(f"**Jumlah Parameter:** {model.count_params():,}")
        except:
            pass
            
    # Tabs untuk Single vs Batch Input
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (Multi-Input)"])
    
    # ==========================================
    # TAB 1: SINGLE PREDICTION
    # ==========================================
    with tab1:
        st.markdown("### Input Data Individual")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        st.info("**Tips:** Isi data dengan benar untuk mendapatkan prediksi yang akurat.")
        
        # Contoh data
        example_col1, example_col2, example_col3 = st.columns(3)
        with example_col1:
            if st.button("Contoh: Anak Normal", use_container_width=True, key="btn_ex_normal"):
                st.session_state.test_age = 24
                st.session_state.test_birth_weight = 3.2
                st.session_state.test_birth_length = 48.5
                st.session_state.test_body_weight = 12.5
                st.session_state.test_body_length = 85.0
                st.rerun()
        with example_col2:
            if st.button("Contoh: Berisiko Stunting", use_container_width=True, key="btn_ex_stunting"):
                st.session_state.test_age = 30
                st.session_state.test_birth_weight = 2.5
                st.session_state.test_birth_length = 47.0
                st.session_state.test_body_weight = 9.0
                st.session_state.test_body_length = 75.0
                st.rerun()
        with example_col3:
            if st.button("Reset", use_container_width=True, key="btn_reset"):
                if 'test_age' in st.session_state:
                    del st.session_state.test_age
                st.rerun()
        
        with col1:
            sex_input = st.selectbox("Jenis Kelamin", ["Male", "Female"], key="single_sex")
            age_input = st.number_input("Umur (bulan)", min_value=0, max_value=120, 
                                    value=st.session_state.get('test_age', 24), key="single_age")
            birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.0, max_value=10.0, 
                                      value=st.session_state.get('test_birth_weight', 3.2), step=0.1, key="single_bw")
            birth_length = st.number_input("Panjang Lahir (cm)", min_value=0.0, max_value=100.0, 
                                      value=st.session_state.get('test_birth_length', 48.5), step=0.1, key="single_bl")
        
        with col2:
            body_weight = st.number_input("Berat Badan Saat Ini (kg)", min_value=0.0, max_value=50.0, 
                                      value=st.session_state.get('test_body_weight', 12.5), step=0.1, key="single_cur_bw")
            body_length = st.number_input("Panjang Badan Saat Ini (cm)", min_value=0.0, max_value=150.0, 
                                      value=st.session_state.get('test_body_length', 85.0), step=0.1, key="single_cur_bl")
        asi_input = st.selectbox("ASI Eksklusif", ["Yes", "No"], key="single_asi")
        
        st.markdown("---")
        
        if st.button("Prediksi Stunting", type="primary", use_container_width=True, key="btn_predict_single"):
            input_data = preprocess_input(
                sex_input, age_input, birth_weight, birth_length,
                body_weight, body_length, asi_input
            )
            
            try:
                # Cek apakah model adalah sklearn atau keras
                if hasattr(model, 'predict_proba'):
                    # sklearn model
                    prediction = model.predict_proba(input_data)
                else:
                    # keras model
                    prediction = model.predict(input_data, verbose=0)
                prob_no_stunting, prob_stunting, result = interpret_prediction(prediction)
                
                st.markdown("### Hasil Prediksi")
                st.markdown("---")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Prediksi", result)
                    fig = px.bar(
                        x=['Tidak Stunting', 'Stunting'],
                        y=[prob_no_stunting, prob_stunting],
                        color=['Tidak Stunting', 'Stunting'],
                        color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
                        labels={'x': 'Status', 'y': 'Probabilitas'},
                        title='Probabilitas Prediksi'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with res_col2:
                    st.subheader("Detail Probabilitas")
                    st.metric("Tidak Stunting", f"{prob_no_stunting*100:.2f}%")
                    st.metric("Stunting", f"{prob_stunting*100:.2f}%")
                    st.progress(float(prob_stunting), text=f"Risiko Stunting: {prob_stunting*100:.1f}%")
                
                st.markdown("---")
                st.subheader("Rekomendasi")
                if result == "Stunting":
                    st.error("""
                    **Anak berisiko stunting. Rekomendasi:**
                    - Konsultasi dengan dokter spesialis anak
                    - Perbaikan gizi dan pola makan
                    - Monitoring pertumbuhan berkala
                    - Pastikan ASI eksklusif jika masih bayi
                    """)
                else:
                    st.success("""
                    **Anak tidak berisiko stunting.**
                    - Tetap jaga pola makan dan gizi seimbang
                    - Lakukan monitoring rutin
                    - Pastikan asupan nutrisi tercukupi
                    """)
            
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

    # ==========================================
    # TAB 2: BATCH PREDICTION
    # ==========================================
    with tab2:
        st.markdown("### Upload Data (CSV)")
        st.write("Unggah file CSV berisi data banyak anak untuk diprediksi sekaligus.")
        
        # Template download
        template_data = {
            "Jenis Kelamin": ["Male", "Female", "Male"],
            "Umur (bulan)": [24, 30, 12],
            "Berat Lahir (kg)": [3.2, 2.5, 3.0],
            "Panjang Lahir (cm)": [48.5, 47.0, 49.0],
            "Berat Badan (kg)": [12.5, 9.0, 10.0],
            "Panjang Badan (cm)": [85.0, 75.0, 78.0],
            "ASI Eksklusif": ["Yes", "No", "Yes"]
        }
        df_template = pd.DataFrame(template_data)
        csv_template = df_template.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Template CSV",
            data=csv_template,
            file_name="template_prediksi_stunting.csv",
            mime="text/csv",
            help="Gunakan template ini untuk memastikan format kolom benar."
        )
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview Data:")
                st.dataframe(df_upload.head())
                
                # Validation basic columns
                required_cols = ["Jenis Kelamin", "Umur (bulan)", "Berat Lahir (kg)", 
                                 "Panjang Lahir (cm)", "Berat Badan (kg)", 
                                 "Panjang Badan (cm)", "ASI Eksklusif"]
                
                missing_cols = [c for c in required_cols if c not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"Format salah! Kolom hilang: {', '.join(missing_cols)}")
                else:
                    if st.button("Mulai Prediksi Batch", type="primary", use_container_width=True):
                        results = []
                        progress_bar = st.progress(0)
                        
                        total_rows = len(df_upload)
                        
                        # Process row by row (safe but slower)
                        # Could be optimized by vectorizing preprocess_input, but complex due to feature engineering logic
                        for idx, row in df_upload.iterrows():
                            # Extract data
                            sex = row["Jenis Kelamin"]
                            age = row["Umur (bulan)"]
                            bw_birth = row["Berat Lahir (kg)"]
                            bl_birth = row["Panjang Lahir (cm)"]
                            bw_curr = row["Berat Badan (kg)"]
                            bl_curr = row["Panjang Badan (cm)"]
                            asi = row["ASI Eksklusif"]
                            
                            # Preprocess
                            processed_input = preprocess_input(
                                sex, age, bw_birth, bl_birth, bw_curr, bl_curr, asi
                            )
                            
                            # Predict
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict_proba(processed_input)
                            else:
                                pred = model.predict(processed_input, verbose=0)
                            
                            p_no, p_yes, res_str = interpret_prediction(pred)
                            
                            results.append({
                                "Index": idx + 1,
                                "Prediksi": res_str,
                                "Probabilitas Stunting (%)": round(p_yes * 100, 2),
                                "Probabilitas Tidak Stunting (%)": round(p_no * 100, 2)
                            })
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        # Combine results
                        df_results = pd.DataFrame(results)
                        df_final = pd.concat([df_upload.reset_index(drop=True), df_results.drop(columns=["Index"])], axis=1)
                        
                        st.success("Prediksi Selesai!")
                        st.dataframe(df_final)
                        
                        # Download result
                        csv_result = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Hasil Prediksi",
                            data=csv_result,
                            file_name="hasil_prediksi_stunting.csv",
                            mime="text/csv"
                        )
                        
                        # Summary Metrics
                        st.markdown("### Ringkasan Hasil")
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        total = len(df_final)
                        stunting_count = len(df_final[df_final["Prediksi"] == "Stunting"])
                        normal_count = len(df_final[df_final["Prediksi"] == "Tidak Stunting"])
                        
                        col_sum1.metric("Total Sampel", total)
                        col_sum2.metric("Terindikasi Stunting", stunting_count, delta_color="inverse")
                        col_sum3.metric("Normal / Tidak Stunting", normal_count)
                        
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")

