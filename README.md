# Ecommerce Customer Churn Analysis and Prediction

## Business Problem Statement

### Case Background

**Perusahaan XYZ** merupakan perusahaan e-commerce yang menyediakan platform transaksi online bagi pelanggan individu dengan beragam kategori produk, metode pembayaran, serta pengalaman belanja digital yang terintegrasi melalui aplikasi dan website. Pelanggan dapat dengan mudah melakukan pencarian produk, memesan barang, memanfaatkan promo, hingga melakukan pembayaran secara fleksibel sesuai preferensi masing-masing.

Sebagai perusahaan yang beroperasi di industri e-commerce yang sangat kompetitif, Perusahaan XYZ menghadapi tantangan besar dalam mempertahankan pelanggan. Konsumen e-commerce cenderung memiliki tingkat loyalitas yang rendah karena banyaknya pilihan platform dengan penawaran harga, promo, dan pengalaman pengguna yang serupa. Persaingan ini mendorong pelanggan untuk berpindah ke kompetitor ketika merasa tidak puas terhadap layanan, harga, maupun pengalaman aplikasi.

**Customer churn** atau hilangnya pelanggan menjadi salah satu tantangan utama. Berdasarkan data internal Perusahaan XYZ, terlihat adanya penurunan aktivitas transaksi pada segmen pelanggan yang sebelumnya aktif dan loyal. Kondisi ini mengindikasikan potensi churn yang dapat berdampak langsung pada penurunan pendapatan serta hilangnya nilai pelanggan jangka panjang (*Customer Lifetime Value*).

Berbagai studi industri juga menunjukkan bahwa biaya memperoleh pelanggan baru jauh lebih tinggi dibandingkan mempertahankan pelanggan yang sudah ada. Penelitian oleh **Rudd et al. (2022)** menyebutkan bahwa biaya akuisisi pelanggan baru dapat mencapai **5–6 kali lipat** dibandingkan biaya retensi. Oleh karena itu, strategi retensi pelanggan menjadi semakin krusial.

---

### Problem Statement

Perusahaan XYZ saat ini **belum memiliki sistem prediktif untuk mengidentifikasi pelanggan yang berisiko churn secara dini**. Meskipun tersedia data historis pelanggan yang mencakup perilaku transaksi, penggunaan aplikasi, serta tingkat kepuasan dan komplain, data tersebut belum dimanfaatkan secara optimal untuk tujuan prediksi.

Akibatnya, strategi retensi yang dijalankan masih bersifat reaktif dan tidak terfokus. Tim bisnis tidak dapat memprioritaskan pelanggan dengan risiko churn tinggi, sehingga intervensi sering kali terlambat atau tidak tepat sasaran. Kondisi ini berpotensi menyebabkan hilangnya pelanggan bernilai tinggi serta meningkatnya biaya akuisisi pelanggan baru.

---

### Goals

- **Mengidentifikasi faktor-faktor utama** yang memengaruhi customer churn
- **Membangun model machine learning** untuk memprediksi pelanggan berisiko churn secara dini
- **Mendukung pengambilan keputusan bisnis yang proaktif** dalam strategi retensi pelanggan

---

### Limitasi Model

1. Data tidak bersifat time series sehingga belum menangkap dinamika perilaku pelanggan dari waktu ke waktu  
2. Ketidakseimbangan kelas churn meskipun telah ditangani dengan teknik resampling  
3. Fitur bersifat indikator tidak langsung terhadap penyebab churn  
4. Definisi churn bersifat prediktif, bukan konfirmasi berhenti permanen  
5. Nilai ekonomi pelanggan belum diperhitungkan dalam pemodelan  

---

## Analytical Approach

Pendekatan yang digunakan adalah **machine learning berbasis klasifikasi** untuk memprediksi pelanggan berisiko churn berdasarkan pola historis perilaku dan pengalaman pelanggan.

Beberapa algoritma dikembangkan dan dibandingkan, termasuk:
- Logistic Regression
- Random Forest
- Gradient Boosting–based models

Karena churn merupakan kelas minoritas yang kritis secara bisnis, dilakukan **penanganan class imbalance** serta evaluasi model difokuskan pada **F2-Score**, yang memberikan bobot lebih besar pada *recall* untuk meminimalkan *False Negative*.

---

## Target Definition

- **Churn (1)**  
  Pelanggan diprediksi tidak lagi melakukan transaksi pada periode waktu tertentu di masa mendatang.

- **Non-Churn (0)**  
  Pelanggan diprediksi masih aktif dan melakukan transaksi pada periode yang sama.

---

## Evaluation Metric

### Confusion Matrix

| ACTUAL / PREDICTED | Not Churn | Churn |
|--------------------|-----------|-------|
| Not Churn | True Negative (TN) | False Positive (FP) |
| Churn | False Negative (FN) | True Positive (TP) |

### Business Interpretation of Errors

- **False Positive (Type 1 Error)**  
  Pelanggan diprediksi churn namun sebenarnya tidak. Dampaknya berupa pemborosan biaya retensi.

- **False Negative (Type 2 Error)**  
  Pelanggan diprediksi tidak churn padahal sebenarnya churn. Dampaknya sangat kritis karena menyebabkan kehilangan pelanggan bernilai tinggi.

### Primary Metric: F2-Score

F2-score digunakan karena memberikan bobot lebih besar pada **recall**, sehingga model difokuskan untuk menangkap sebanyak mungkin pelanggan churn.

\[
F2 = \frac{(1 + 2^2) \cdot Precision \cdot Recall}{(2^2 \cdot Precision) + Recall}
\]

---

## Data Preparation & Feature Engineering

### Drop Column
- `CustomerID` dihapus karena hanya bersifat identifier dan tidak relevan untuk prediksi.

### Data Splitting
- Train–test split **80:20**
- Stratified sampling untuk menjaga proporsi churn
- Random state untuk reproducibility

### Encoding & Scaling
- **One Hot Encoding** untuk fitur kategorikal nominal
- **RobustScaler** untuk fitur numerik agar tahan terhadap outlier

---

## Modelling Strategy

### Model Candidates
- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- LightGBM  

### Handling Imbalanced Data
- SMOTE  
- Random OverSampler  
- Random UnderSampler  
- NearMiss  

Setiap model diuji dengan berbagai teknik resampling menggunakan **Stratified K-Fold Cross Validation**.

---

## Model Evaluation Results

### Best Performing Models
- **XGBoost + SMOTE**
- **LightGBM + RandomOverSampler**

### Final Model
**LightGBM Tuned + RandomOverSampler**

| Metric | Score |
|------|-------|
| F2-Score (Test) | **0.9716** |
| Recall (Test) | **0.9737** |
| Precision (Test) | **0.9635** |

Model menunjukkan performa tinggi dan stabil dengan generalisasi yang baik.

---

## Model Interpretability

Fitur paling berpengaruh terhadap churn:
- **Tenure**
- **Complain**
- **CashbackAmount**
- **WarehouseToHome**
- **DaySinceLastOrder**
- **OrderCount**

Analisis SHAP menunjukkan bahwa:
- Pelanggan baru dan pelanggan dengan komplain memiliki risiko churn tinggi
- Penurunan aktivitas dan pengalaman logistik buruk menjadi sinyal awal churn

---

## Business Impact Analysis

### Assumsi Biaya
- Akuisisi pelanggan baru: **$5,000**
- Program retensi pelanggan: **$1,000**

### Tanpa Model
- Total biaya: **$950,000**

### Dengan Model
- Pelanggan ditarget retensi: 192  
- Pelanggan churn tidak terdeteksi: 5  

**Total biaya: $217,000**

### Cost Saving
**$733,000** (≈ 77% penghematan biaya)

---

##  Deploy Model to Web Base

Deployment Machine Learning adalah proses membuat model machine learning yang telah dilatih (trained) dapat digunakan dalam aplikasi yang dapat diakses oleh pengguna akhir secara online.

Deployment machine learning memiliki peran penting dalam pemanfaatan model machine learning secara optimal dalam berbagai aplikasi yang dapat mempermudah dan mempercepat pengambilan keputusan.

1. Exploratory Data Analyst

<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.47.10.png" />
<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.47.22.png" />
<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.47.41.png" />



2. Prediksi Churn
   
<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.47.53.png" />
<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.48.08.png" />
<img width="2940" height="1912" alt="image" src="https://github.com/ulfahrsdn/Alpha-Group---JCDSOL-022/blob/main/assets/Screenshot%202026-01-29%20at%2022.48.19.png" />

---

## Conclusion

Model **LightGBM Tuned + RandomOverSampler** terbukti mampu mendeteksi pelanggan berisiko churn secara akurat, stabil, dan selaras dengan kebutuhan bisnis. Dengan pendekatan berbasis data ini, Perusahaan XYZ dapat beralih dari strategi retensi yang reaktif menjadi **preventif, efisien, dan berorientasi pada nilai bisnis jangka panjang**.

Link Colab:  <br>
Link Tableu: https://public.tableau.com/app/profile/ulfah.rosdiana/viz/Remedial_FinalProject/Dashboard3?publish=yes <br>
Link download material: https://www.kaggle.com/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction <br>
Link demo / deployment: https://alphafinprojectchurnanalysisprediction.streamlit.app <br>

