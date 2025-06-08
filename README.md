# Laporan Proyek Prediksi Dropout Mahasiswa

## 1. Latar Belakang

Tingkat dropout mahasiswa menjadi perhatian utama dalam dunia pendidikan tinggi. Prediksi dini terhadap mahasiswa yang berisiko dropout sangat penting agar institusi dapat melakukan intervensi yang tepat untuk meningkatkan retensi dan keberhasilan akademik. Proyek ini bertujuan mengembangkan model prediksi dropout menggunakan data historis mahasiswa dan menyajikan hasilnya dalam dashboard bisnis serta aplikasi web interaktif.

---

## 2. Data dan Label Encoding

Dataset terdiri dari beberapa fitur mahasiswa, termasuk status akademik yang dikategorikan menjadi tiga kelas:

- Graduate (Lulus)
- Enrolled (Sedang Terdaftar)
- Dropout (Putus Studi)

Kode berikut digunakan untuk mengubah label status menjadi angka agar dapat diproses model machine learning:

```python
status_mapping = {"Graduate": 0, "Enrolled": 1, "Dropout": 2}
df["Status"] = df["Status"].map(status_mapping)
```

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

Dashboard bisnis dibuat menggunakan Looker Studio (Google Data Studio) untuk menampilkan **hasil prediksi status mahasiswa (Graduate, Enrolled, Dropout)** berdasarkan probabilitas persen hasil prediksi. Tujuannya adalah untuk monitoring status mahasiswa yang sedang enrolled dengan detail probabilitas dropout.

Link Dashboard:
[https://lookerstudio.google.com/reporting/147758c6-e4b2-4611-9481-3943b33c3c0b](https://lookerstudio.google.com/reporting/147758c6-e4b2-4611-9481-3943b33c3c0b)

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

---

## 10. Hasil Temuan dari Data Enrolled

Dari data mahasiswa yang berstatus **Enrolled** sebanyak 794 mahasiswa, prediksi model terhadap risiko dropout menunjukkan:

- **444 mahasiswa (56%) diprediksi tidak akan dropout**
- **347 mahasiswa (44%) diprediksi berpotensi dropout**

---

## 11. Rekomendasi

Berdasarkan hasil prediksi tersebut, berikut rekomendasi untuk institusi pendidikan:

- **Fokus Intervensi pada Mahasiswa Berisiko Dropout:**
  Mahasiswa yang diprediksi berpotensi dropout (347 orang) perlu mendapatkan perhatian khusus berupa pendampingan akademik, bimbingan karir, dan dukungan psikologis.

- **Monitoring Mahasiswa Enrolled:**
  Dashboard probabilitas dropout harus digunakan secara rutin untuk mengawasi perubahan risiko mahasiswa yang sedang terdaftar.

- **Analisis Faktor Penyebab:**
  Lakukan kajian mendalam terhadap faktor-faktor yang berkontribusi tinggi pada dropout berdasarkan fitur penting dari model, guna mengembangkan kebijakan dan program pencegahan yang efektif.

- **Peningkatan Kualitas Data:**
  Perbaiki pengumpulan data dan standar input agar model prediksi dapat semakin akurat dan reliabel.

---

## 12. Kesimpulan

- Model Random Forest berhasil memprediksi risiko dropout mahasiswa dengan akurasi tinggi (94%) dan performa seimbang antara recall dan precision.
- Aplikasi Streamlit memudahkan pengguna dalam mengakses prediksi dan mengambil keputusan berbasis data secara langsung.
- Dashboard bisnis menampilkan hasil prediksi probabilitas dropout mahasiswa enrolled secara transparan untuk monitoring dan intervensi.
- Rekomendasi intervensi ditujukan agar institusi dapat menekan angka dropout dan meningkatkan tingkat kelulusan.

---
