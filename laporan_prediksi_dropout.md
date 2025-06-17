# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech Dengan Prediksi Dropout Mahasiswa

---

## 1. Business Understanding

### Latar Belakang Bisnis

Sebagai institusi pendidikan tinggi, perusahaan atau universitas perlu memastikan tingkat kelulusan yang tinggi dan meminimalkan angka dropout mahasiswa. Dropout mahasiswa tidak hanya berdampak pada reputasi institusi, tapi juga berimbas pada aspek finansial dan sumber daya. Dengan meningkatnya jumlah mahasiswa baru setiap tahun, penting bagi institusi untuk memiliki sistem prediksi dini risiko dropout agar dapat melakukan intervensi yang tepat dan personal.

### Permasalahan Bisnis

- Tidak adanya sistem otomatis yang dapat memprediksi mahasiswa yang berisiko dropout sejak awal masa studi.
- Sulitnya mengidentifikasi mahasiswa berisiko tinggi di antara ribuan mahasiswa aktif.
- Kurangnya data yang terintegrasi dan analisis berbasis data untuk mendukung keputusan manajemen dalam hal retensi mahasiswa.
- Keterbatasan alat yang memudahkan pihak manajemen untuk memonitor status mahasiswa secara real-time dan mengantisipasi potensi dropout.

### Cakupan Proyek

Proyek ini akan membangun sebuah model prediksi untuk mengidentifikasi mahasiswa yang berisiko dropout menggunakan data historis mahasiswa. Selain itu, akan dibangun dashboard bisnis untuk menampilkan hasil prediksi secara visual berdasarkan probabilitas, dan aplikasi web berbasis Streamlit untuk kemudahan interaksi pengguna dalam melakukan prediksi secara mandiri. Cakupan proyek meliputi:

- Pengolahan data dan pelabelan status mahasiswa
- Pelatihan model klasifikasi menggunakan Random Forest
- Pembuatan dashboard visualisasi hasil prediksi
- Pengembangan aplikasi Streamlit untuk prediksi interaktif
- Rekomendasi bisnis berdasarkan hasil analisis

### Setup Environment

**Disarankan menggunakan virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

**Instalasi Library:**

```bash
pip install -r requirements.txt
```

---

## 2. Data dan Label Encoding

### Persiapan

Sumber data: [https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset ini mencakup berbagai fitur demografis, akademik, dan keuangan mahasiswa, dengan status akhir mahasiswa dikategorikan menjadi tiga kelas:

- **Graduate** (Lulus)
- **Enrolled** (Sedang Terdaftar)
- **Dropout** (Putus Studi)

Label dikodekan menjadi angka untuk keperluan pemodelan:

```python
status_mapping = {"Graduate": 0, "Enrolled": 1, "Dropout": 2}
df["Status"] = df["Status"].map(status_mapping)
```

### Temuan EDA: Top 10 Fitur Berkorelasi Terhadap Dropout

Analisis korelasi menunjukkan bahwa variabel-variabel berikut memiliki tingkat hubungan tertinggi terhadap status mahasiswa, khususnya dalam membedakan mahasiswa yang dropout dan yang tidak.

| Peringkat | Fitur                             | Korelasi terhadap Status | Interpretasi Utama                                                                            |
| --------- | --------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| 1         | Curricular_units_2nd_sem_approved | **0.654**                | Mahasiswa yang menyelesaikan lebih banyak mata kuliah di semester 2 cenderung lulus.          |
| 2         | Curricular_units_2nd_sem_grade    | **0.605**                | Nilai tinggi di semester 2 berkaitan erat dengan kelulusan.                                   |
| 3         | Curricular_units_1st_sem_approved | **0.555**                | Semakin banyak mata kuliah semester 1 yang lulus, semakin kecil kemungkinan dropout.          |
| 4         | Curricular_units_1st_sem_grade    | **0.520**                | Prestasi akademik awal menjadi indikator kuat terhadap keberlangsungan studi.                 |
| 5         | Tuition_fees_up_to_date           | **0.442**                | Mahasiswa yang membayar tepat waktu lebih kecil risikonya untuk dropout.                      |
| 6         | Scholarship_holder                | **0.313**                | Penerima beasiswa cenderung lebih bertahan dalam studi.                                       |
| 7         | Age_at_enrollment                 | **-0.267**               | Mahasiswa yang lebih tua saat masuk cenderung memiliki risiko dropout yang lebih tinggi.      |
| 8         | Debtor                            | **-0.267**               | Mahasiswa dengan utang cenderung lebih berisiko untuk dropout.                                |
| 9         | Gender                            | **-0.252**               | Korelasi negatif menunjukkan adanya perbedaan risiko antara gender (perlu eksplorasi lanjut). |
| 10        | Application_mode                  | **-0.245**               | Mode pendaftaran tertentu mungkin memengaruhi risiko dropout.                                 |

> **Catatan:** Korelasi positif artinya fitur tersebut cenderung meningkat jika status menuju “Graduate”, sedangkan korelasi negatif menunjukkan kecenderungan ke arah “Dropout”.

**Insight Utama dari EDA Ini:**

- Faktor akademik merupakan indikator paling kuat terkait risiko dropout.
- Faktor finansial (seperti keterlambatan pembayaran dan status beasiswa) juga signifikan.
- Faktor demografis seperti usia dan jenis kelamin memiliki korelasi yang lebih rendah, namun tetap relevan.

---

## 3. Pemisahan Data

- Data mahasiswa dengan status **Enrolled** dipisahkan untuk kebutuhan dashboard dan monitoring (tidak digunakan untuk pelatihan model):

  ```python
  df_enrolled = df[df["Status"] == 1].copy()
  ```

- Data mahasiswa dengan status **Graduate** dan **Dropout** digunakan sebagai data pelatihan model klasifikasi:

  ```python
  df_train = df[df["Status"].isin([0, 2])].copy()
  df_train["Dropout_Flag"] = df_train["Status"].apply(lambda x: 1 if x == 2 else 0)
  ```

---

## 4. Pemilihan Fitur (Feature Selection)

Fitur dipisahkan menjadi dua kelompok:

- **Fitur Kategorikal** seperti status pernikahan, mode pendaftaran, jenis kursus, kualifikasi orang tua, dsb.
- **Fitur Numerikal** seperti usia saat pendaftaran, nilai masuk, jumlah mata kuliah yang diambil, dan indikator ekonomi makro (tingkat pengangguran, inflasi, GDP).

```python
categorical_features = [ ... ]
numerical_features = [ ... ]
```

---

## 5. Preprocessing dan Pipeline Model

Dibuat pipeline preprocessing dengan dua langkah utama:

- Untuk fitur kategorikal: imputasi nilai kosong dengan modus (most frequent) dan encoding one-hot.
- Untuk fitur numerikal: imputasi nilai kosong dengan rata-rata.

```python
categorical_transformer = Pipeline([...])
numerical_transformer = Pipeline([...])
preprocessor = ColumnTransformer([...])
```

Kemudian pipeline lengkap dibuat dengan menambahkan model klasifikasi Random Forest dengan 100 estimator dan penyesuaian class weight agar seimbang.

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(...))
])
```

---

## 6. Pelatihan Model dan Evaluasi

Data dibagi menjadi training dan testing (80%:20%) dengan stratifikasi label dropout agar distribusi tetap seimbang.

```python
X_train, X_test, y_train, y_test = train_test_split(..., stratify=y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
```

Evaluasi menggunakan classification report:

| Kelas             | Precision | Recall | F1-Score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| Tidak Dropout (0) | 0.93      | 0.96   | 0.95     | 442     |
| Dropout (1)       | 0.94      | 0.89   | 0.92     | 284     |

- **Akurasi keseluruhan:** 94%
- Model menunjukkan performa baik, terutama dalam memprediksi mahasiswa yang berpotensi dropout.

---

## 7. Penyimpanan Model

Model yang telah dilatih disimpan menggunakan joblib agar dapat digunakan kembali:

```python
joblib.dump(model, "model/dropout_model.pkl")
```

---

## 8. Business Dashboard

### Tools yang Digunakan:

- **Google Looker Studio (Google Data Studio)**

### Link Dashboard:

[Dashboard Dropout Mahasiswa – Looker Studio](https://lookerstudio.google.com/reporting/147758c6-e4b2-4611-9481-3943b33c3c0b)

### Tujuan Dashboard:

Dashboard ini dirancang untuk membantu pihak manajemen pendidikan memonitor mahasiswa yang sedang aktif (status _Enrolled_) berdasarkan hasil prediksi probabilitas dropout. Dengan tampilan visual yang informatif, dashboard ini memungkinkan pengambilan keputusan yang lebih cepat dan berbasis data dalam merancang intervensi preventif.

---

### Fitur Visualisasi:

1. **Distribusi performa mahasiswa berdasarkan status**:

   - Visualisasi 4 fitur terpenting (`grade` dan `approved` per semester).
   - Memberikan gambaran umum karakteristik mahasiswa dropout.

2. **Filter interaktif berdasarkan probabilitas dropout, grade dan approve per semester**.

3. **Pie chart prediksi Enrolled**:

   - Untuk 794 mahasiswa Enrolled:

     - 447 (56.3%) diprediksi tidak akan dropout
     - 347 (43.7%) diprediksi berisiko dropout

---

## 9. Aplikasi Web Streamlit

Aplikasi interaktif dibuat dengan Streamlit agar pengguna dapat mengunggah data CSV mahasiswa secara langsung dan mendapatkan prediksi dropout secara real-time.

### Fitur utama aplikasi:

- Upload file CSV dengan delimiter `;`
- Pembersihan data kolom nilai yang tidak valid (misalnya format angka dengan koma, titik, dsb)
- Validasi kelengkapan kolom sesuai kebutuhan model
- Menampilkan hasil prediksi dan probabilitas dropout setiap mahasiswa
- Ringkasan metrik jumlah mahasiswa dan potensi dropout
- Download hasil prediksi dalam format CSV

Link aplikasi:
[https://prediksidropout-jych3hraetercxm6xmprmg.streamlit.app/](https://prediksidropout-jych3hraetercxm6xmprmg.streamlit.app/)

### Cara Menjalankan Secara Lokal:

```bash
streamlit run app.py
```

---

## 10. Conclusion

Model prediksi yang dikembangkan menggunakan Random Forest mampu mencapai **akurasi 94%** dan **F1-score 0.92** untuk kelas dropout, menandakan performa yang baik dan seimbang. Dari data mahasiswa yang berstatus **Enrolled** sebanyak 794 mahasiswa, prediksi model terhadap risiko dropout menunjukkan:

- **444 mahasiswa (56%) diprediksi tidak akan dropout**
- **347 mahasiswa (44%) diprediksi berpotensi dropout**

---

## 11. Rekomendasi Action Items

### **Action Item 1 – Intervensi Akademik untuk Mahasiswa Berisiko**

Berikan program bimbingan belajar dan pendampingan akademik secara rutin kepada mahasiswa yang memiliki nilai semester pertama rendah dan jumlah mata kuliah disetujui sedikit.

---

### **Action Item 2 – Perbaikan dan Dukungan Keuangan Mahasiswa**

Sistem pembayaran uang kuliah yang fleksibel, pengingat berkala, serta peningkatan distribusi beasiswa kepada mahasiswa yang menunjukkan potensi akademik tetapi rentan secara ekonomi.

---

## 10. Business Dashboard

### Tools:

Google Looker Studio

### Link:

[Dashboard Dropout Mahasiswa](https://lookerstudio.google.com/reporting/147758c6-e4b2-4611-9481-3943b33c3c0b)

### Fitur Visualisasi:

1. **Distribusi performa mahasiswa berdasarkan status**:

   - Visualisasi 4 fitur terpenting (`grade` dan `approved` per semester).
   - Memberikan gambaran umum karakteristik mahasiswa dropout.

2. **Filter interaktif berdasarkan probabilitas, grade dan approve setiap semester**.

3. **Pie chart prediksi Enrolled**:

   - Untuk 794 mahasiswa Enrolled:

     - 444 (56%) diprediksi tidak akan dropout
     - 347 (44%) diprediksi berisiko dropout

### Insight:

- Mahasiswa dengan **nilai semester pertama buruk cenderung dropout**.
- **Pembayaran tidak tepat waktu** sangat berkorelasi dengan risiko DO.
- **Penerima beasiswa memiliki peluang kelulusan yang lebih tinggi**.
