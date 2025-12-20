"""Fungsi untuk model dan prediksi"""
import streamlit as st
import os
import numpy as np
from constants import MODEL_PATH

# Import untuk model
try:
    from tensorflow import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    keras = None


@st.cache_resource
def load_model(model_path):
    """Load model H5"""
    if not MODEL_AVAILABLE:
        return None
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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

