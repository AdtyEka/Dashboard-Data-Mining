"""Fungsi untuk model dan prediksi"""
import streamlit as st
import os
import numpy as np
from constants import MODEL_PATH

# Import untuk model
try:
    from tensorflow import keras
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    keras = None
    tf = None


def _check_h5_structure(model_path):
    """Cek struktur file HDF5 untuk menentukan apakah hanya berisi weights"""
    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            keys = list(f.keys())
            # Model lengkap biasanya punya 'model_config' atau 'config'
            # Weights saja biasanya hanya punya layer names atau 'model_weights'
            has_config = 'model_config' in keys or 'config' in keys
            has_weights = 'model_weights' in keys or any('weight' in str(k).lower() for k in keys)
            return has_config, has_weights, keys
    except Exception:
        return None, None, []


@st.cache_resource
def load_model(model_path):
    """Load model H5 dengan berbagai metode fallback"""
    if not MODEL_AVAILABLE:
        return None
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ File model {model_path} tidak ditemukan.")
        return None
    
    errors = []
    
    # Cek struktur file HDF5
    has_config, has_weights, h5_keys = _check_h5_structure(model_path)
    
    # Metode 1: Load model lengkap (default)
    try:
        model = keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Model berhasil dimuat menggunakan metode default")
        return model
    except Exception as e1:
        errors.append(f"Metode 1 (default): {str(e1)}")
    
    # Metode 2: Load dengan safe_mode=False (untuk TensorFlow 2.16+)
    try:
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        st.success(f"✅ Model berhasil dimuat menggunakan safe_mode=False")
        return model
    except Exception as e2:
        errors.append(f"Metode 2 (safe_mode=False): {str(e2)}")
    
    # Metode 3: Load dengan custom_objects kosong (untuk model tanpa custom objects)
    try:
        model = keras.models.load_model(model_path, compile=False, custom_objects={})
        st.success(f"✅ Model berhasil dimuat dengan custom_objects kosong")
        return model
    except Exception as e3:
        errors.append(f"Metode 3 (custom_objects): {str(e3)}")
    
    # Metode 4: Coba dengan tf.keras langsung
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Model berhasil dimuat menggunakan tf.keras")
        return model
    except Exception as e4:
        errors.append(f"Metode 4 (tf.keras): {str(e4)}")
    
    # Metode 5: Coba load dengan skip_mismatch jika hanya weights
    if has_weights and not has_config:
        try:
            # Jika hanya weights, kita perlu membuat arsitektur model dulu
            # Tapi karena kita tidak tahu arsitekturnya, kita coba metode lain
            st.warning("⚠️ File hanya berisi weights, bukan model lengkap. Mencoba metode alternatif...")
        except Exception as e5:
            errors.append(f"Metode 5 (weights only): {str(e5)}")
    
    # Metode 6: Coba dengan allow_pickle=True (untuk model lama)
    try:
        model = keras.models.load_model(model_path, compile=False, custom_objects=None)
        st.success(f"✅ Model berhasil dimuat")
        return model
    except Exception as e6:
        errors.append(f"Metode 6 (allow_pickle): {str(e6)}")
    
    # Jika semua metode gagal, tampilkan error yang lebih informatif
    error_msg = f"""
**Error loading model: {model_path}**

**Detail Error:**
{chr(10).join(f"- {err}" for err in errors)}

**Struktur File HDF5:**
- Keys ditemukan: {', '.join(h5_keys) if h5_keys else 'Tidak dapat dibaca'}
- Memiliki config: {has_config if has_config is not None else 'Tidak diketahui'}
- Memiliki weights: {has_weights if has_weights is not None else 'Tidak diketahui'}

**Kemungkinan penyebab:**
1. File model corrupt atau tidak valid
2. Versi TensorFlow/Keras tidak kompatibel (versi saat ini: {tf.__version__ if tf else 'Tidak terdeteksi'})
3. Model disimpan dengan format yang berbeda
4. Model hanya berisi weights tanpa config (arsitektur model hilang)
5. Model dibuat dengan versi TensorFlow yang berbeda

**Solusi:**
1. **Pastikan model disimpan dengan lengkap:**
   ```python
   model.save('best_stunting_model.h5')  # ✅ Benar
   # Bukan: model.save_weights('best_stunting_model.h5')  # ❌ Salah
   ```

2. **Update TensorFlow:**
   ```bash
   pip install --upgrade tensorflow>=2.13.0
   ```

3. **Jika model hanya berisi weights:**
   - Anda perlu membuat ulang arsitektur model yang sama
   - Kemudian load weights: `model.load_weights('best_stunting_model.h5')`

4. **Cek kompatibilitas versi:**
   - Pastikan versi TensorFlow yang digunakan untuk load sama dengan versi saat training
   - Atau coba dengan versi TensorFlow yang lebih baru/lebih lama

5. **Alternatif: Simpan ulang model dengan format yang lebih kompatibel:**
   ```python
   # Jika Anda punya akses ke kode training
   model.save('best_stunting_model.h5', save_format='h5')
   # Atau gunakan SavedModel format
   model.save('best_stunting_model_savedmodel')
   ```
    """
    st.error(error_msg)
    return None


def preprocess_input(sex, age, birth_weight, birth_length, body_weight, body_length, asi):
    """Preprocess input untuk prediksi model"""
    # Normalisasi
    age_max = 60.0
    birth_weight_max = 4.5
    birth_length_max = 55.0
    body_weight_max = 20.0
    body_length_max = 110.0
    
    age_norm = float(age) / age_max if age_max > 0 else 0.0
    birth_weight_norm = float(birth_weight) / birth_weight_max if birth_weight_max > 0 else 0.0
    birth_length_norm = float(birth_length) / birth_length_max if birth_length_max > 0 else 0.0
    body_weight_norm = float(body_weight) / body_weight_max if body_weight_max > 0 else 0.0
    body_length_norm = float(body_length) / body_length_max if body_length_max > 0 else 0.0
    
    # Feature engineering
    age_years = float(age) / 12.0
    bmi = float(body_weight) / ((float(body_length) / 100.0) ** 2) if body_length > 0 else 0.0
    weight_growth = float(body_weight) - float(birth_weight)
    length_growth = float(body_length) - float(birth_length)
    weight_per_age = float(body_weight) / float(age) if age > 0 else 0.0
    length_per_age = float(body_length) / float(age) if age > 0 else 0.0
    
    # Binary features
    low_birth_weight = 1.0 if birth_weight < 2.5 else 0.0
    short_birth_length = 1.0 if birth_length < 48.0 else 0.0
    asi_numerical = 1.0 if asi == "Yes" else 0.0
    asi_weight_growth = weight_growth if asi == "Yes" else 0.0
    nutritional_stress = (1.0 - body_weight_norm) * (1.0 - body_length_norm)
    log_body_weight = np.log(float(body_weight) + 1.0) if body_weight > 0 else 0.0
    
    features = [
        age_norm, birth_weight_norm, birth_length_norm, body_weight_norm, body_length_norm,
        age_years, bmi, weight_growth, length_growth, weight_per_age, length_per_age,
        low_birth_weight, short_birth_length, asi_numerical, asi_weight_growth,
        nutritional_stress, log_body_weight
    ]
    
    return np.array([features], dtype=np.float32)


def interpret_prediction(prediction):
    """Interpretasi hasil prediksi model"""
    if len(prediction[0]) == 1:
        prob_stunting = max(0.0, min(1.0, float(prediction[0][0])))
        prob_no_stunting = 1.0 - prob_stunting
    else:
        prob_no_stunting = float(prediction[0][0])
        prob_stunting = float(prediction[0][1]) if len(prediction[0]) > 1 else 0.0
        total = prob_no_stunting + prob_stunting
        if total > 0:
            prob_no_stunting /= total
            prob_stunting /= total
    
    threshold = 0.5
    result = "Stunting" if prob_stunting > threshold else "Tidak Stunting"
    return prob_no_stunting, prob_stunting, result

