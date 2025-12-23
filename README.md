# MALLORN: Phân Loại Tidal Disruption Event (TDE)
##  1. Tổng quan

**TDE (Tidal Disruption Event)** là sự kiện khi một ngôi sao bị xé rách bởi siêu lỗ đen.  
Chúng cực kỳ hiếm (**~100 sự kiện đã quan sát**) → class imbalance nặng (~5% là TDE).

**Nhiệm vụ:**
- Phân loại nhị phân: **TDE (1)** vs **non-TDE (0)**
- **Metric:** F1-score
- Dữ liệu: multi-band light curves + metadata

---

##  2. Dữ liệu

| Tập | Số object | Số TDE | Quan sát |
|-----|-----------|---------|--------------|
| Train | 3,043 | 148 (4.9%) | 479,384 |
| Test | 7,135 | ? | 1,145,125 |

Mỗi object gồm:
- LC trong 6 filter: **u, g, r, i, z, y**
- Metadata: `Z` (redshift), `EBV` (extinction)
- 20 splits dạng `split_01 → split_20`

---

##  3. Tóm tắt giải pháp

### Feature Engineering — **366 features/object**

**Gaussian Process smoothing (GP-RBF)**  
- flux_smooth  
- derivative & smoothness  
- peak-slope  
- variability metrics  

**Thống kê cơ bản**  
- mean, std, skew, kurtosis  
- percentiles (5–95%)  
- RMS, MAD, IQR  

**Thời gian & hình dạng light curve**  
- rise/decline rate  
- peak timing  
- early/peak/late phases  

**Color features (đa band)**  
- g–r, r–i, g–i  
- u–g (UV excess → dấu hiệu TDE)  
- Δ(g–r) theo pha  

**Power-law decay fitting**  
- TDE thường có α ≈ 1.3–1.8  
- thêm `decay_alpha`, `decay_rms`

**Tương tác redshift**  
- log1p(z), z², EBV × Z

---

## 4. Mô hình

Sử dụng 4 mô hình chính:

| Model | Lý do chọn |
|-------|---------|
| XGBoost | Vì lightcurve nhiều biến động khiến modal dễ overfit nên chọn XGBoost để khắc phục |
| LightGBM | 366 features rất nặng cần mô hình học nhanh  |
| RandomForest | Ổn định, không nhạy với nhiễu, Tạo diversity cho ensemble |
| CatBoost | Flux thay đổi thất thường, các band có thể thiếu (u/g/r không đủ điểm) |

### ✔ Ensemble theo công thức:
P_final = wx·XGB + wl·LGBM + wr·RF + wc·CAT

###Thresholad tối ưu tìm được: 0.217

## 5. Kết quả

| Model        | CV F1  | Precision | Recall | Threshold |
|--------------|--------|-----------|--------|-----------|
| XGBoost      | 0.412  | 0.39      | 0.44   | 0.19      |
| LightGBM     | 0.428  | 0.41      | 0.45   | 0.18      |
| RandomForest | 0.351  | 0.33      | 0.38   | 0.21      |
| CatBoost     | 0.401  | 0.38      | 0.42   | 0.20      |
| **Ensemble** | **0.472** | **0.45** | **0.50** | **0.217** |

- **Public Leaderboard:** 0.5833
- **Private Leaderboard:** 0.5694  

