# 🌌 MALLORN: Tidal Disruption Event Classification

<div align="center">

**Phân loại sự kiện sao bị xé rách bởi lỗ đen siêu khối lượng**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![F1 Score](https://img.shields.io/badge/F1%20Score-0.5833-brightgreen.svg)](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

</div>

---

## 👥 Team Information

**Nhóm 9 - INT3405E_4 - Trường Đại học Công nghệ, ĐHQGHN**

| Thành viên | MSSV |
|------------|------|
| Nguyễn Khánh Tùng | 23021713 |
| Phạm Việt Hà | 23021541 |
| Đinh Công Minh | 23021625 |

---

## 📋 Table of Contents

- [Tổng quan](#-tổng-quan)
- [Dữ liệu](#-dữ-liệu)
- [Phương pháp](#-phương-pháp)
- [Kết quả](#-kết-quả)

---

## 🎯 Tổng quan

### Giới thiệu về TDE

**Tidal Disruption Event (TDE)** là hiện tượng thiên văn hiếm gặp xảy ra khi một ngôi sao đi quá gần lỗ đen siêu khối lượng và bị lực triều xé rách. Các mảnh vỡ của ngôi sao tạo thành đĩa bồi tụ phát ra bức xạ mạnh, đặc biệt trong dải UV và X-ray.

### Thách thức

- **Độ hiếm:** Chỉ ~100 TDE đã được quan sát trong vũ trụ
- **Class imbalance nghiêm trọng:** ~5% TDE vs 95% non-TDE
- **Dữ liệu phức tạp:** Multi-band time-series với noise cao, missing values

### Mục tiêu

Xây dựng mô hình phân loại nhị phân:
- **Positive class (1):** Tidal Disruption Event  
- **Negative class (0):** Non-TDE (Supernovae, AGN, variable stars, etc.)
- **Evaluation metric:** F1-score (balanced precision-recall cho imbalanced data)

---

## 📊 Dữ liệu

### Dataset Statistics

| Split | Objects | TDE | Non-TDE | Observations | TDE Ratio |
|-------|---------|-----|---------|--------------|-----------|
| **Train** | 3,043 | 148 | 2,895 | 479,384 | 4.9% |
| **Test** | 7,135 | ? | ? | 1,145,125 | Unknown |

### Data Structure

#### Light Curves (`*_full_lightcurves.csv`)
- **object_id**: Định danh duy nhất
- **Time (MJD)**: Ngày Julian sửa đổi (Modified Julian Date)
- **Flux**: Giá trị đo flux trắc quang
- **Flux_err**: Độ không đảm bảo của Flux (sai số)
- **Filter**: Dải trắc quang (Photometric band) {u, g, r, i, z, y}

#### Metadata (`train_log.csv`, `test_log.csv`)
- **object_id**: Liên kết với light curves
- **target**: Nhãn nhị phân (chỉ có trong tập train)
- **Z**: Redshift (khoảng cách vũ trụ học)
- **EBV**: Độ tắt quang E(B-V) (do bụi Ngân Hà)

### Data Characteristics

- **Multi-band time series:** 6 bộ lọc trắc quang trải dài từ UV đến NIR
- **Irregular sampling:** Các khoảng thời gian lấy mẫu không đều nhau
- **Missing data:** Không phải tất cả thiên thể đều được quan sát ở mọi band
- **High noise:** Dữ liệu thiên văn chứa các sai số đo lường

---

## 🔬 Phương pháp

### Pipeline Overview

```
Raw Light Curves → Preprocessing → Feature Engineering → Model Training → Ensemble → Prediction
```

### 1. Preprocessing

#### Extinction Correction
Hiệu chỉnh ảnh hưởng của bụi Milky Way:

$$F_{\text{corrected}} = F_{\text{observed}} \times 10^{0.4 \times R_\lambda \times E(B-V)}$$

Với $R_\lambda$ là hệ số extinction cho từng band:
- u: 4.81 | g: 3.64 | r: 2.70 | i: 2.06 | z: 1.58 | y: 1.31

### 2. Feature Engineering (366 features)

#### A. Gaussian Process Smoothing
- **Kernel:** RBF + WhiteKernel
- **Output:** Flux đã làm mượt, đạo hàm, ước lượng phương sai
- **Purpose:** Khử nhiễu time-series và nội suy các khoảng trống dữ liệu

#### B. Statistical Features (per band)
| Danh mục | Features | Số lượng |
|----------|----------|-------|
| Xu hướng tập trung | mean, median, weighted_mean | 3 |
| Độ phân tán | std, MAD, IQR, RMS | 4 |
| Dạng phân phối | skewness, kurtosis | 2 |
| Phân vị | 5th, 25th, 75th, 95th | 4 |
| Cực trị | min, max, range | 3 |

#### C. Temporal Features (Đặc trưng thời gian)
- **Rise/decline rates:** Tốc độ tăng/giảm flux
- **Peak timing:** Thời điểm đạt peak flux so với cửa sổ quan sát
- **Phase-based stats:** Đặc trưng giai đoạn đầu/đỉnh/cuối (Early/peak/late)
- **Variability indices:** Chỉ số Chi-squared, Stetson

#### D. Color Features (Liên kết các band)
Chênh lệch magnitude giữa các band (u-g, g-r, r-i, g-i):
- TDE có **UV excess** (lượng dư UV) → u-g nhỏ hơn các hiện tượng thoáng qua (transients) khác
- Color evolution (tiến hóa màu sắc) theo thời gian: Δ(g-r), Δ(r-i)

#### E. Power-law Decay Fitting
TDE decay thường tuân theo quy luật: $F(t) \propto t^{-\alpha}$ với $\alpha \approx 1.3$-1.8

Fit và extract:
- `decay_alpha`: Chỉ số Power-law
- `decay_rms`: Độ khớp (Goodness of fit)
- `t_half`: Thời gian giảm một nửa độ sáng (Half-light time)

#### F. Metadata Interactions
- $\log(1+z)$, $z^2$: Biến đổi Redshift
- $\text{EBV} \times z$: Kết hợp Extinction-redshift
- Band-specific extinction corrections

### 3. Model Architecture

#### Base Models

| Model | Hyperparameters | Lý do chọn |
|-------|----------------|-----------|
| **XGBoost** | `n_estimators=500`<br>`max_depth=7`<br>`learning_rate=0.05`<br>`scale_pos_weight=19` | Xử lý overfitting với regularization<br>Cân bằng dữ liệu với weight |
| **LightGBM** | `n_estimators=500`<br>`num_leaves=63`<br>`learning_rate=0.05` | Training nhanh cho 366 features<br>Sử dụng bộ nhớ hiệu quả |
| **RandomForest** | `n_estimators=300`<br>`max_depth=15`<br>`class_weight='balanced'` | Bền vững với outliers<br>Tạo sự đa dạng cho ensemble |
| **CatBoost** | `iterations=500`<br>`depth=6`<br>`learning_rate=0.05` | Xử lý missing values tốt<br>Cấu trúc cây đối xứng |

#### Ensemble Strategy

**Weighted Soft Voting:**

$$P_{\text{final}} = w_1 \cdot P_{\text{XGB}} + w_2 \cdot P_{\text{LGBM}} + w_3 \cdot P_{\text{RF}} + w_4 \cdot P_{\text{CAT}}$$

Weights được optimize qua cross-validation để maximize F1-score.

#### Threshold Optimization

**Optimal threshold:** 0.217 (vs default 0.5)

### 4. Cross-validation Strategy

- **Method:** Stratified 5-Fold CV
- **Stratification:** Giữ nguyên TDE ratio trong mỗi fold
- **Evaluation:** F1-score trung bình qua các folds

---

## 📈 Kết quả

### Model Performance

| Model | CV F1 ↑ | Precision | Recall | Optimal Threshold |
|--------------|---------|-----------|--------|-------------------|
| XGBoost | 0.5249 | 0.50 | 0.55 | 0.19 |
| LightGBM | 0.5288 | 0.51 | 0.55 | 0.18 |
| RandomForest | 0.5111 | 0.48 | 0.54 | 0.21 |
| CatBoost | 0.3820 | 0.36 | 0.41 | 0.20 |
| **Ensemble (Weighted)** | **0.5307** | **0.52** | **0.56** | **0.217** |

### Competition Results

| Metric | Score |
|--------|-------|
| **Public Leaderboard F1** | **0.5833** |
| Cross-validation F1 | 0.5307 |
| Improvement over CV | +9.9% |

### Feature Importance (Top 10)

*Kết quả từ ensemble model aggregation*

1. `g_peak_flux` - Peak flux trong g-band
2. `u_mean_flux` - Flux trung bình dải UV (Dấu hiệu đặc trưng của TDE)
3. `r_rise_rate` - Tốc độ tăng flux
4. `color_ug` - Chỉ thị lượng dư UV (UV excess)
5. `z_redshift_log` - Khoảng cách vũ trụ học
6. `g_gp_smoothness` - Độ mượt từ GP
7. `ri_color_evolution` - Sự thay đổi màu sắc theo thời gian
8. `i_peak_timing` - Vị trí tương đối của peak
9. `decay_alpha` - Chỉ số Power-law
10. `g_skewness` - Độ lệch phân phối flux

---


---

## 🔍 Key Insights

### Điều gì làm TDE khác biệt?

1. **UV Excess:** TDE có u-band flux cao hơn đáng kể so với supernovae.
2. **Smooth Light Curves:** Ít biến động thất thường hơn so với các đợt bùng phát AGN.
3. **Characteristic Decay:** Tuân theo luật lũy thừa (Power-law) với $\alpha \sim 1.3$-$1.8$.
4. **Blue Colors:** Màu u-g và g-r thấp hơn ("xanh hơn") so với các hiện tượng transient đỏ.
5. **Rise Time:** Tăng flux chậm hơn so với một số loại supernovae.

### Thách thức của mô hình

- **Extreme Imbalance:** Chỉ 4.9% là positive class.
- **Feature Overlap:** TDE và một số SNe có các đặc điểm tương tự nhau gây nhầm lẫn.
- **Missing Data:** Không phải tất cả objects đều có đủ 6 bands.
- **Noise:** Các phép đo thiên văn vốn dĩ có độ không đảm bảo (uncertainty).

---

### Competition
- [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

### Tools & Libraries
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [CatBoost](https://catboost.ai/)
- [scikit-learn](https://scikit-learn.org/)

---




