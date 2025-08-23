# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech Dengan Prediksi Dropout Mahasiswa

---

## 1. Business Understanding

### Latar Belakang

Sebagai institusi pendidikan tinggi, universitas atau perusahaan edutech memiliki tanggung jawab untuk memastikan mahasiswa dapat menyelesaikan studinya dengan baik. Namun, kenyataannya, tidak sedikit mahasiswa yang mengalami **dropout** atau **putus studi** di tengah jalan. Tingginya angka dropout berdampak langsung pada:

- **Reputasi akademik** institusi,
- **Kinerja finansial**, baik dari sisi biaya pendidikan maupun pengelolaan beasiswa,
- **Alokasi sumber daya manusia**, seperti dosen pembimbing dan layanan akademik.

Dalam era digital dengan jumlah mahasiswa yang semakin besar, pengambilan keputusan berbasis intuisi tidak lagi memadai. Diperlukan pendekatan **berbasis data** untuk memetakan risiko dropout secara lebih objektif, cepat, dan terukur.

---

### Permasalahan yang Dihadapi

Beberapa tantangan utama yang ingin diselesaikan melalui proyek ini meliputi:

- Belum adanya **sistem deteksi dini** yang mampu mengidentifikasi mahasiswa berisiko dropout pada tahap awal studi.
- **Sulitnya mengawasi ribuan mahasiswa** aktif secara individual oleh pihak manajemen kampus.
- **Kurangnya visualisasi interaktif dan monitoring real-time** terhadap kinerja mahasiswa, terutama untuk mereka yang masih dalam status _Enrolled_.
- Minimnya **analisis berbasis performa akademik semester awal** untuk mendukung pengambilan keputusan intervensi yang tepat sasaran.

---

### Tujuan dan Solusi yang Ditawarkan

Proyek ini bertujuan untuk membantu institusi pendidikan dalam menekan angka dropout melalui penerapan **model prediktif** dan **dashboard analitik**. Solusi yang dikembangkan mencakup:

1. **Model klasifikasi dropout**: Menggunakan algoritma _Random Forest_ yang dilatih dari data historis mahasiswa untuk memprediksi apakah seorang mahasiswa berisiko tinggi dropout atau tidak.
2. **Dashboard interaktif berbasis Looker Studio**:

   - Visualisasi performa akademik mahasiswa berdasarkan fitur kunci seperti **nilai dan jumlah mata kuliah yang disetujui per semester**.
   - Fitur filter untuk memantau mahasiswa _Enrolled_ dengan **probabilitas dropout tinggi** secara real-time.

3. **Aplikasi web interaktif** berbasis Streamlit:

   - Memungkinkan pengguna mengunggah data baru dan langsung memperoleh hasil prediksi secara praktis.
   - Memberikan metrik agregat dan hasil ekspor prediksi untuk keperluan manajerial.

---

### Cakupan Proyek

Untuk menjawab tantangan tersebut, proyek ini mencakup beberapa tahapan utama berikut:

- **Eksplorasi dan pra-pemrosesan data** mahasiswa dari berbagai latar belakang akademik dan sosiodemografis.
- **Pelabelan dan pembentukan target klasifikasi dropout** berdasarkan status akhir studi mahasiswa.
- **Pelatihan model prediksi** menggunakan algoritma _Random Forest_ dengan penyesuaian class weight.
- **Pengembangan dashboard bisnis** untuk memvisualisasikan hasil prediksi dan insight performa mahasiswa.
- **Pembuatan aplikasi Streamlit** agar pengguna non-teknis dapat memanfaatkan model secara mandiri.
- **Penyusunan rekomendasi kebijakan** berbasis data untuk mendukung strategi retensi mahasiswa.

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

### Tools yang Digunakan

- **Google Looker Studio (Google Data Studio)**
  Platform ini dipilih karena kemampuannya menampilkan visualisasi data interaktif yang mudah digunakan oleh pihak manajemen non-teknis.

---

### Link Dashboard

📊 [Dashboard Dropout Mahasiswa – Looker Studio](https://lookerstudio.google.com/reporting/147758c6-e4b2-4611-9481-3943b33c3c0b)

---

### Tujuan Dashboard

Dashboard ini dirancang untuk membantu institusi pendidikan dalam:

- Memonitor mahasiswa aktif (_Enrolled_) berdasarkan probabilitas prediksi dropout.
- Mengidentifikasi mahasiswa dengan risiko tinggi secara visual dan cepat.
- Memberikan insight berbasis data untuk pengambilan keputusan intervensi dini.

Dashboard juga dapat digunakan untuk **evaluasi internal kualitas akademik** berdasarkan kinerja semester awal mahasiswa.

---

### Struktur Dashboard dan Fitur Visualisasi

Dashboard terdiri dari **2 halaman utama**:

---

#### 🟦 **Halaman 1: Approve Analysis**

Visualisasi fokus pada performa mahasiswa dari segi jumlah mata kuliah yang **disetujui (approved)**.

**Fitur utama:**

- **Pie Chart Hasil Prediksi Dropout** menggambarkan proporsi hasil klasifikasi model terhadap mahasiswa Enrolled.
- **Rata-rata jumlah mata kuliah yang disetujui** di Semester 1 dan Semester 2, dibedakan berdasarkan kategori prediksi dropout (Dropout vs Tidak Dropout).
- **Diagram batang per siswa**: membandingkan jumlah mata kuliah yang disetujui di Semester 1 dan Semester 2 untuk setiap mahasiswa.
- **Filter interaktif**:

  - Berdasarkan hasil prediksi dropout (Dropout / Tidak Dropout).
  - Berdasarkan status akademik aktual (Graduate, Enrolled, Dropout).
  - Berdasarkan **probabilitas prediksi dropout ≥ 51%** untuk fokus pada risiko tinggi.

---

#### 🟦 **Halaman 2: Grade Analysis**

Visualisasi difokuskan pada **nilai rata-rata mahasiswa (grade)** per semester.

**Fitur utama:**

- **Pie Chart Hasil Prediksi Dropout** menggambarkan proporsi hasil klasifikasi model terhadap mahasiswa Enrolled.
- **Rata-rata nilai (grade)** Semester 1 dan Semester 2 untuk mahasiswa dropout dan tidak.
- **Diagram batang per siswa**: perbandingan nilai masing-masing mahasiswa antar semester.
- Memberikan gambaran konsistensi atau penurunan performa dari semester 1 ke 2.
- **Filter interaktif**:

  - Berdasarkan kategori hasil prediksi (Dropout / Tidak Dropout).
  - Berdasarkan status akademik aktual (Enrolled, Graduate, Dropout).
  - Berdasarkan **threshold probabilitas dropout**.

---

### Insight yang Didapat dari Dashboard

- Mahasiswa yang diprediksi dropout umumnya memiliki nilai dan jumlah mata kuliah yang disetujui lebih rendah, baik di semester 1 maupun semester 2.
- Perbandingan performa antar semester dapat digunakan untuk mengidentifikasi pola penurunan performa akademik.
- Fitur filter memudahkan manajemen untuk fokus pada kelompok risiko tinggi dan merancang strategi intervensi berbasis bukti.

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

Berikut adalah versi **revisi dan perluasan** dari bagian **Conclusion** dan **Actionable Recommendations** untuk benar-benar menjawab permasalahan institusi serta menjelaskan faktor penyebab dropout dan karakteristik mahasiswa yang dropout, sesuai dengan catatan reviewer:

---

## 10. Conclusion: Analisis dan Temuan Utama

Model prediksi yang dikembangkan menggunakan algoritma **Random Forest** menunjukkan performa tinggi dalam mengklasifikasi risiko dropout mahasiswa:

- **Akurasi keseluruhan: 94%**
- **F1-score kelas dropout: 0.92**
- Model berhasil **mengidentifikasi dengan baik mahasiswa yang berisiko tinggi mengalami dropout**.

### Karakteristik Umum Mahasiswa Dropout

Berdasarkan analisis fitur dan eksplorasi data (EDA), mahasiswa yang mengalami dropout cenderung memiliki pola sebagai berikut:

1. **Nilai akademik rendah** di semester pertama dan kedua, terutama dalam bentuk rata-rata nilai mata kuliah (_grade_).
2. **Jumlah mata kuliah yang disetujui (approved) rendah**, baik di semester pertama maupun kedua.
3. Cenderung mengalami stagnasi atau penurunan performa antara semester 1 dan semester 2.
4. Beberapa berasal dari latar belakang sosial-ekonomi tertentu, meskipun faktor akademik lebih dominan secara statistik.

### Faktor-Faktor Terpenting yang Berpengaruh pada Dropout

Berdasarkan analisis _feature importance_ dari model:

| Fitur                               | Pengaruh (Importance) |
| ----------------------------------- | --------------------- |
| `Curricular_units_2nd_sem_approved` | 14.5%                 |
| `Curricular_units_2nd_sem_grade`    | 13.8%                 |
| `Curricular_units_1st_sem_approved` | 9.9%                  |
| `Curricular_units_1st_sem_grade`    | 8.4%                  |

Fitur-fitur ini menunjukkan bahwa performa akademik dua semester pertama sangat menentukan risiko dropout. Mahasiswa yang gagal mempertahankan jumlah SKS yang disetujui dan mendapatkan nilai baik sangat rentan untuk tidak melanjutkan studi.

---

## 11. Actionable Recommendations: Langkah Konkret untuk Institusi

### 1. Intervensi Akademik Terarah Berdasarkan Prediksi

- Gunakan dashboard probabilitas dropout untuk **mengidentifikasi mahasiswa Enrolled yang berisiko tinggi (≥51%)**.
- Terapkan **program remedial, bimbingan belajar, atau tutor sebaya** secara aktif untuk mahasiswa dengan _grade_ rendah dan _approved_ sedikit.
- Bangun sistem notifikasi dini berbasis data semester awal untuk mahasiswa baru.

---

### 2. Monitoring Berbasis Dashboard dan Interaksi Real-Time

- Manfaatkan dashboard Looker Studio untuk memonitor:

  - **Perbandingan nilai dan approved semester 1 vs semester 2**.
  - **Distribusi risiko berdasarkan kategori prediksi dan status akademik**.

- Gunakan filter interaktif untuk **mengelompokkan mahasiswa berdasarkan kategori risiko** dan melakukan analisis kelompok (cohort) secara berkala.

---
