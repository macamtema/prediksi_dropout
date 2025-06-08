import streamlit as st

# ✅ HARUS jadi Streamlit command pertama
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")

import pandas as pd
import numpy as np
import joblib

# Fungsi untuk membersihkan nilai grade yang tidak valid
def clean_grade(value):
    try:
        return float(value)
    except:
        try:
            val_str = str(value).strip().replace(",", ".")
            if val_str.endswith('.'):
                val_str = val_str[:-1]
            return float(val_str)
        except:
            try:
                val_str = str(value).replace(".", "")
                return float(val_str) / (10 ** (len(val_str) - 2))
            except:
                return np.nan

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    return joblib.load("model/dropout_model.pkl")

model = load_model()

# --- Antarmuka Aplikasi ---
st.markdown("## 🎓 Prediksi Dropout Mahasiswa")
st.markdown("Unggah file CSV mahasiswa untuk memprediksi kemungkinan **dropout akademik** berdasarkan fitur yang tersedia.")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload File CSV Mahasiswa", type="csv")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, delimiter=';')
        st.success("✅ File berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")
        st.stop()

    # Tambahkan kolom ID jika belum ada
    if 'ID' not in data.columns:
        data.insert(0, 'ID', range(1, len(data) + 1))

    # Kolom nilai yang butuh pembersihan
    grade_columns = [
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_grade",
        "Admission_grade", "Unemployment_rate", "Inflation_rate", "GDP"
    ]

    for col in grade_columns:
        if col in data.columns:
            data[col] = data[col].apply(clean_grade)
            if data[col].isnull().any():
                data[col].fillna(data[col].mean(), inplace=True)

    # Validasi kolom yang dibutuhkan oleh model
    try:
        categorical_features = model.named_steps["preprocessor"].transformers[0][2]
        numerical_features = model.named_steps["preprocessor"].transformers[1][2]
        expected_columns = list(categorical_features) + list(numerical_features)
    except Exception as e:
        st.error(f"❌ Model tidak memiliki struktur preprocessor yang sesuai: {e}")
        st.stop()

    missing_cols = [col for col in expected_columns if col not in data.columns]
    if missing_cols:
        st.error(f"❌ File CSV tidak memiliki kolom berikut yang diperlukan: {missing_cols}")
        st.stop()

    # Prediksi
    try:
        preds = model.predict(data[expected_columns])
        pred_proba = model.predict_proba(data[expected_columns])[:, 1]
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
        st.stop()

    # Tambahkan hasil prediksi ke data
    data["Dropout_Prediction"] = np.where(preds == 1, "Dropout", "Tidak Dropout")
    data["Probabilitas_Dropout (%)"] = (pred_proba * 100).round(2)

    # Ringkasan metrik
    st.markdown("### 📊 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Mahasiswa", len(data))
    with col2:
        st.metric("Potensi Dropout", int((preds == 1).sum()))

    # Tabel hasil prediksi
    st.dataframe(
        data[["ID", "Dropout_Prediction", "Probabilitas_Dropout (%)"] + [col for col in data.columns if col not in ['ID', 'Dropout_Prediction', 'Probabilitas_Dropout (%)']]],
        use_container_width=True,
        height=500
    )

    # Tombol download
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu untuk memulai prediksi.")
