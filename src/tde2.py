!unzip -q "/content/drive/MyDrive/mallorn-astronomical-classification-challenge.zip" -d /content/data

!pip install catboost
"""
MALLORN TDE CLASSIFICATION (FINAL)

Competition: MALLORN Tidal Disruption Event Classification

Architecture:
- 366 đặc trưng được trích xuất từ multi-band light curves (u,g,r,i,z,y)
- 4 model ensemble: XGBoost + LightGBM + RandomForest + CatBoost
- Weighted averaging với optimized threshold

"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from catboost import CatBoostClassifier

# SECTION 1: DATA LOADING
# Load 20 train/test splits và merge vào 1 dataframes


def load_all_lightcurves(base_path='/content/data/'):
    """
    Load và nối 20 train light curve splits.

    Returns:
        DataFrame với các cột: object_id, Time (MJD), Flux, Flux_err, Filter
    """
    all_lightcurves = []
    for i in range(1, 21):
        split_name = f'split_{i:02d}'
        file_path = f'{base_path}{split_name}/train_full_lightcurves.csv'
        try:
            df = pd.read_csv(file_path)
            all_lightcurves.append(df)
            print(f"  Loaded {split_name}")
        except FileNotFoundError:
            print(f'  Warning: {file_path} not found')
    return pd.concat(all_lightcurves, ignore_index=True)

def load_test_lightcurves(base_path='/content/data/'):
    """Load và nối 20 test light curve splits."""
    all_test = []
    for i in range(1, 21):
        try:
            df = pd.read_csv(f'{base_path}split_{i:02d}/test_full_lightcurves.csv')
            all_test.append(df)
        except:
            continue
    return pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()

print("\n[1/6] Loading training data...")
train_log = pd.read_csv('/content/data/train_log.csv')
train_lightcurves = load_all_lightcurves()

print("\n[2/6] Loading test data...")
test_log = pd.read_csv('/content/data/test_log.csv')
test_lightcurves = load_test_lightcurves()

print(f"\nData load thành công:")
print(f"  Train objects: {len(train_log):,}")
print(f"  Test objects: {len(test_log):,}")
print(f"  Train observations: {len(train_lightcurves):,}")
print(f"  Test observations: {len(test_lightcurves):,}")

# SECTION 2: HIỆU CHỈNH EXTINCTION
# Apply Milky Way dust extinction correction using E(B-V) values

def apply_extinction_correction(flux, ebv, filter_name):

    # Extinction coefficients for each filter (R_λ values)
    R = {
        'u': 4.81,  # U-band bị ảnh hưởng nhất bởi bụi
        'g': 3.64,
        'r': 2.70,
        'i': 2.06,
        'z': 1.58,
        'y': 1.31   # Y-band ít bị nhất
    }

    # Calculate extinction in magnitudes: A_λ = R_λ × E(B-V)
    A_lambda = R.get(filter_name, 0) * ebv

    # Convert to flux correction: F_corrected = F_observed × 10^(0.4 × A_λ)
    return flux * 10**(0.4 * A_lambda)

# Merge E(B-V) values vào light curves
print("\n[3/6] Applying extinction correction...")
train_lightcurves = train_lightcurves.merge(
    train_log[['object_id', 'EBV']],
    on='object_id',
    how='left'
)
test_lightcurves = test_lightcurves.merge(
    test_log[['object_id', 'EBV']],
    on='object_id',
    how='left'
)

# Apply correction to all observations
train_lightcurves['Flux_corrected'] = train_lightcurves.apply(
    lambda r: apply_extinction_correction(r['Flux'], r['EBV'], r['Filter']),
    axis=1
)
test_lightcurves['Flux_corrected'] = test_lightcurves.apply(
    lambda r: apply_extinction_correction(r['Flux'], r['EBV'], r['Filter']),
    axis=1
)

print("  Extinction correction complete")

# ====================================================================
# STEP 2.5 — GAUSSIAN PROCESS SMOOTHING + UNIFORM RESAMPLING
# ====================================================================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def gp_smooth_and_resample(time, flux, flux_err, dt=1.0):
    """
    Fit Gaussian Process:
      kernel = RBF(length_scale) + WhiteKernel(noise_level)
    Resample lightcurve to uniform cadence (dt days).
    Return:
      t_grid, flux_smooth, flux_var, flux_derivative
    """
    if len(time) < 3:
        return None

    t = np.array(time)
    y = np.array(flux)
    yerr = np.array(flux_err)

    sort_idx = np.argsort(t)
    t, y, yerr = t[sort_idx], y[sort_idx], yerr[sort_idx]

    kernel = 1.0 * RBF(length_scale=np.median(np.diff(t)) + 1e-3) + \
             WhiteKernel(noise_level=np.mean(yerr)**2)

    try:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=yerr**2, normalize_y=True)
        gp.fit(t.reshape(-1, 1), y)

        t_grid = np.arange(t.min(), t.max(), dt)

        y_mean, y_std = gp.predict(t_grid.reshape(-1, 1), return_std=True)
        
        dy = np.gradient(y_mean, dt)

        return t_grid, y_mean, y_std, dy
    except:
        return None

# SECTION 3: FEATURE ENGINEERING
# Extract 366 features per object based on MALLORN paper
# Features include:
#   - Statistical aggregations (mean, std, percentiles, etc.)
#   - Time-series characteristics (peak timing, rise/decline rates)
#   - Multi-band colors (g-r, r-i, g-i)
#   - Color evolution across light curve phases
#   - Signal-to-noise metrics
#   - Power-law decay fitting (α parameter)
# ============================================================================

def extract_advanced_lightcurve_features(lc_group):
    """
    Extract comprehensive features from a single-band light curve.

    Args:
        lc_group: DataFrame with Time (MJD), Flux_corrected, Flux_err columns

    Returns:
        Series with ~60 features per band
    """
    features = {}

    # Extract arrays và sort theo tgian
    time = lc_group['Time (MJD)'].values
    flux = lc_group['Flux_corrected'].values
    flux_err = lc_group['Flux_err'].values

    # Require ít nhất 3 observations để số liệu có nghĩa
    if len(flux) < 3:
        return pd.Series(features)
    
        # ------------------------------------------------------------
    # 3.12: GAUSSIAN PROCESS SMOOTHING + RESAMPLED FEATURES
    # ------------------------------------------------------------
    gp_result = gp_smooth_and_resample(time, flux, flux_err, dt=1.0)

    if gp_result is not None:
        t_grid, smooth, smooth_std, deriv = gp_result

        # GP smoothed stats
        features['gp_flux_mean'] = np.mean(smooth)
        features['gp_flux_std'] = np.std(smooth)
        features['gp_flux_max'] = np.max(smooth)
        features['gp_flux_min'] = np.min(smooth)

        # GP variability
        features['gp_amp'] = (smooth.max() - smooth.min()) / (smooth.max() + smooth.min() + 1e-10)
        features['gp_rms'] = np.sqrt(np.mean(smooth**2))

        # GP derivative features
        features['gp_deriv_mean'] = np.mean(deriv)
        features['gp_deriv_std'] = np.std(deriv)
        features['gp_deriv_max'] = np.max(deriv)
        features['gp_deriv_min'] = np.min(deriv)

        # Slope near peak (t +/- 5 days)
        if len(t_grid) > 5:
            mid = len(t_grid) // 2
            window = 5
            s = slice(max(0, mid-window), min(len(t_grid), mid+window))
            features['gp_peak_slope'] = np.mean(deriv[s])
        else:
            features['gp_peak_slope'] = 0

        # Smoothness (variance of derivative)
        features['gp_smoothness'] = 1.0 / (1e-6 + np.var(deriv))

    else:
        # Fill zeros if GP failed
        gp_keys = [
            'gp_flux_mean','gp_flux_std','gp_flux_max','gp_flux_min',
            'gp_amp','gp_rms','gp_deriv_mean','gp_deriv_std',
            'gp_deriv_max','gp_deriv_min','gp_peak_slope','gp_smoothness'
        ]
        for k in gp_keys:
            features[k] = 0


    # Sort theo tgian (important for time-series features)
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    # 3.1: BASIC STATISTICAL FEATURES
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_median'] = np.median(flux)
    features['flux_max'] = np.max(flux)
    features['flux_min'] = np.min(flux)
    features['flux_range'] = features['flux_max'] - features['flux_min']

    # Higher-order moments for distribution shape
    features['flux_skew'] = pd.Series(flux).skew()  # bất cân xứng
    features['flux_kurtosis'] = pd.Series(flux).kurtosis()  # nặng đuôi

    # Percentiles for robust statistics
    features['flux_05th'] = np.percentile(flux, 5)
    features['flux_25th'] = np.percentile(flux, 25)
    features['flux_75th'] = np.percentile(flux, 75)
    features['flux_95th'] = np.percentile(flux, 95)

    # Interquartile range (IQR) - robust measure of spread
    features['flux_iqr'] = features['flux_75th'] - features['flux_25th']
    features['flux_iqr_ratio'] = features['flux_iqr'] / (features['flux_median'] + 1e-10)

    # 3.2: TIME COVERAGE FEATURES
    features['duration'] = time.max() - time.min()  # số ngày quan sát
    features['n_observations'] = len(lc_group)  # số lượng đo

    # Sampling cadence statistics
    if len(time) > 1:
        time_diffs = np.diff(time)
        features['mean_time_gap'] = np.mean(time_diffs)
        features['median_time_gap'] = np.median(time_diffs)
        features['std_time_gap'] = np.std(time_diffs)  # tần suất đo
        features['max_time_gap'] = np.max(time_diffs)  # gap lớn nhất
    else:
        features['mean_time_gap'] = features['median_time_gap'] = 0
        features['std_time_gap'] = features['max_time_gap'] = 0

    # 3.3: PEAK-RELATED FEATURES
    # TDEs có tính peak + decay - peak timing là quan trọng
    peak_idx = np.argmax(flux)
    t_peak = time[peak_idx]
    features['peak_flux'] = flux[peak_idx]

    # tgian đến/ từ peak
    features['time_to_peak'] = t_peak - time.min()  # Rise time
    features['time_after_peak'] = time.max() - t_peak  # Decline time
    features['peak_position_ratio'] = features['time_to_peak'] / (features['duration'] + 1e-10)

    # tốc độ rise/ decline
    if peak_idx > 0 and peak_idx < len(flux) - 1:
        features['rise_rate'] = features['peak_flux'] / (features['time_to_peak'] + 1e-10)
        features['decline_rate'] = features['peak_flux'] / (features['time_after_peak'] + 1e-10)
        features['rise_decline_ratio'] = features['rise_rate'] / (features['decline_rate'] + 1e-10)

        # Symmetry: TDEs thường rise nhanh hơn decline
        features['rise_decline_symmetry'] = abs(features['time_to_peak'] - features['time_after_peak']) / (features['duration'] + 1e-10)
    else:
        features['rise_rate'] = features['decline_rate'] = 0
        features['rise_decline_ratio'] = features['rise_decline_symmetry'] = 0

    # 3.4: VARIABILITY FEATURES
    # Amplitude (normalized flux range)
    features['amplitude'] = (features['flux_max'] - features['flux_min']) / (features['flux_max'] + features['flux_min'] + 1e-10)

    # Outlier statistics
    features['beyond_1std'] = np.sum(np.abs(flux - features['flux_mean']) > features['flux_std']) / len(flux)
    features['beyond_2std'] = np.sum(np.abs(flux - features['flux_mean']) > 2*features['flux_std']) / len(flux)

    # Positive flux fraction (negative values can indicate noise)
    features['flux_positive_frac'] = np.sum(flux > 0) / len(flux)

    # 3.5: WEIGHTED STATISTICS (ERROR-WEIGHTED)
    # Weight by inverse variance for more reliable measurements
    weights = 1 / (flux_err**2 + 1e-10)
    features['flux_weighted_mean'] = np.average(flux, weights=weights)
    features['flux_weighted_std'] = np.sqrt(np.average((flux - features['flux_weighted_mean'])**2, weights=weights))

    # Chi-squared per degree of freedom (goodness of fit to constant)
    features['chi2_per_dof'] = np.sum((flux - features['flux_mean'])**2 / (flux_err**2 + 1e-10)) / (len(flux) - 1 + 1e-10)

    # 3.6: FLUX CHANGE FEATURES
    if len(flux) > 1:
        flux_changes = np.diff(flux)
        features['max_flux_change'] = np.max(np.abs(flux_changes))
        features['mean_abs_flux_change'] = np.mean(np.abs(flux_changes))
    else:
        features['max_flux_change'] = features['mean_abs_flux_change'] = 0

    # 3.7: EARLY vs LATE EVOLUTION
    # Compare first half vs second half of observations
    time_split = np.median(time)
    early_mask = time <= time_split
    late_mask = time > time_split

    if early_mask.sum() > 0 and late_mask.sum() > 0:
        features['flux_early_mean'] = np.mean(flux[early_mask])
        features['flux_late_mean'] = np.mean(flux[late_mask])
        features['flux_evolution'] = features['flux_late_mean'] - features['flux_early_mean']
    else:
        features['flux_early_mean'] = features['flux_late_mean'] = features['flux_mean']
        features['flux_evolution'] = 0

    # 3.8: ROBUST STATISTICS
    # Median Absolute Deviation (MAD) - robust alternative to std
    features['mad'] = np.median(np.abs(flux - features['flux_median']))

    # Root Mean Square
    features['flux_rms'] = np.sqrt(np.mean(flux**2))

    # 3.9: SIGNAL-TO-NOISE RATIO (SNR) FEATURES
    snr = flux / (flux_err + 1e-10)
    features['snr_mean'] = np.mean(snr)
    features['snr_median'] = np.median(snr)
    features['snr_max'] = np.max(snr)
    features['snr_std'] = np.std(snr)

    # Fraction of high-SNR measurements
    features['snr_frac_gt3'] = np.mean(snr > 3)
    features['snr_frac_gt5'] = np.mean(snr > 5)
    features['snr_frac_gt10'] = np.mean(snr > 10)

    # 3.10: PHASE-BASED FEATURES
    # chia light curve thành các phases relative to peak
    # TDEs have distinct signatures in each phase

    pre_mask = time < (t_peak - 10)  # Pre-peak (>10 days before)
    near_mask = (time >= (t_peak - 10)) & (time <= (t_peak + 10))  # Near peak
    post_mask = (time > (t_peak + 10)) & (time <= (t_peak + 30))  # Post-peak

    features['n_pre_peak'] = pre_mask.sum()
    features['n_near_peak'] = near_mask.sum()
    features['n_post_peak'] = post_mask.sum()

    # More detailed phase windows
    early_phase = (time >= (t_peak - 10)) & (time < t_peak)  # Rise phase
    peak_phase = (time >= (t_peak - 5)) & (time <= (t_peak + 5))  # Peak plateau
    late_phase = (time > (t_peak + 10)) & (time <= (t_peak + 30))  # Decline

    # Average flux in each phase
    if early_phase.sum() > 0:
        features['flux_early_phase_mean'] = np.mean(flux[early_phase])
    else:
        features['flux_early_phase_mean'] = features['flux_mean']

    if peak_phase.sum() > 0:
        features['flux_peak_phase_mean'] = np.mean(flux[peak_phase])
    else:
        features['flux_peak_phase_mean'] = features['peak_flux']

    if late_phase.sum() > 0:
        features['flux_late_phase_mean'] = np.mean(flux[late_phase])
    else:
        features['flux_late_phase_mean'] = features['flux_mean']

    # 3.11: POWER-LAW DECAY FITTING
    # TDEs thường tuân theo power-law decay: F ∝ t^(-α)
    # Typical TDE: α ≈ 1.3 - 1.8 (van Velzen et al. 2021)

    decay_mask = (time > (t_peak + 5)) & (flux > 0)  # Post-peak decay phase

    if decay_mask.sum() >= 3:  # Need at least 3 points for fit
        t_decay = time[decay_mask] - t_peak  # Time since peak
        f_decay = flux[decay_mask]

        try:
            # Fit log(F) = -α × log(t) + const
            log_t = np.log(t_decay + 0.1)  # Small offset to avoid log(0)
            log_f = np.log(f_decay)

            # Linear least squares in log-log space
            A = np.vstack([log_t, np.ones_like(log_t)]).T
            coef, residuals, _, _ = np.linalg.lstsq(A, log_f, rcond=None)

            features['decay_alpha'] = -coef[0]  # Power-law index (positive value)

            # RMS residuals (quality of fit)
            if len(residuals) > 0:
                features['decay_rms'] = np.sqrt(residuals[0] / len(log_f))
            else:
                features['decay_rms'] = 0
        except:
            features['decay_alpha'] = 0
            features['decay_rms'] = 999  # Flag for bad fit
    else:
        features['decay_alpha'] = 0
        features['decay_rms'] = 999

    return pd.Series(features)

# --------------------------------------------------------------------------------------------------------------------------

def create_advanced_features(lightcurves_df):
    """
    Create full feature matrix from light curves.

    Process:
    1. Extract per-band features (6 bands × ~60 features = 360 features)
    2. Create cross-band features (colors, timing differences)
    3. Create phase-based color evolution features
    4. Create coverage/quality metrics

    Returns:
        DataFrame with one row per object, ~366 feature columns
    """
    features_list = []
    unique_objects = lightcurves_df['object_id'].unique()

    print(f"\nExtracting features from {len(unique_objects)} objects...")

    for idx, obj_id in enumerate(unique_objects):
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(unique_objects)}")

        # Get all observations for this object
        obj_lc = lightcurves_df[lightcurves_df['object_id'] == obj_id]
        obj_features = {'object_id': obj_id}

        # Dictionary to store per-band features
        filter_features = {}

        # 3.12: PER-BAND FEATURE EXTRACTION
        # Extract features separately for each filter band
        for filter_name in ['u', 'g', 'r', 'i', 'z', 'y']:
            filter_lc = obj_lc[obj_lc['Filter'] == filter_name]

            if len(filter_lc) >= 3:  # Minimum observations for meaningful features
                features = extract_advanced_lightcurve_features(filter_lc)
                filter_features[filter_name] = features

                # Add to object features with band prefix
                for feat_name, feat_val in features.items():
                    obj_features[f'{filter_name}_{feat_name}'] = feat_val
            else:
                filter_features[filter_name] = None

        # 3.13: CROSS-BAND COLOR FEATURES
        # Colors are differences in magnitudes between bands
        # TDEs have characteristic color evolution

        # g-r color (most important for TDE identification)
        if filter_features.get('g') is not None and filter_features.get('r') is not None:
            obj_features['color_g_minus_r'] = filter_features['g']['flux_mean'] - filter_features['r']['flux_mean']
            obj_features['flux_ratio_g_over_r'] = filter_features['g']['flux_mean'] / (filter_features['r']['flux_mean'] + 1e-10)

        # r-i color
        if filter_features.get('r') is not None and filter_features.get('i') is not None:
            obj_features['color_r_minus_i'] = filter_features['r']['flux_mean'] - filter_features['i']['flux_mean']
            obj_features['amplitude_ratio_r_i'] = filter_features['r']['amplitude'] / (filter_features['i']['amplitude'] + 1e-10)

        # g-i color (wider baseline)
        if filter_features.get('g') is not None and filter_features.get('i') is not None:
            obj_features['color_g_minus_i'] = filter_features['g']['flux_mean'] - filter_features['i']['flux_mean']

        # Peak timing alignment across bands
        if filter_features.get('g') is not None and filter_features.get('r') is not None:
            obj_features['peak_time_diff_g_r'] = filter_features['g']['time_to_peak'] - filter_features['r']['time_to_peak']

        # 3.14: COLOR EVOLUTION FEATURES
        # TDEs show characteristic color changes over time
        # Convert flux to magnitude: m = -2.5 × log10(F)

        def flux_to_mag(f):
            return -2.5 * np.log10(f + 1e-10)

        # g-r color evolution (early → peak → late)
        if filter_features.get('g') is not None and filter_features.get('r') is not None:
            g_early = filter_features['g'].get('flux_early_phase_mean', 0)
            r_early = filter_features['r'].get('flux_early_phase_mean', 0)
            g_peak = filter_features['g'].get('flux_peak_phase_mean', 0)
            r_peak = filter_features['r'].get('flux_peak_phase_mean', 0)
            g_late = filter_features['g'].get('flux_late_phase_mean', 0)
            r_late = filter_features['r'].get('flux_late_phase_mean', 0)

            # Color at different phases
            if g_early > 0 and r_early > 0:
                obj_features['g_r_color_early'] = flux_to_mag(g_early) - flux_to_mag(r_early)
            if g_peak > 0 and r_peak > 0:
                obj_features['g_r_color_peak'] = flux_to_mag(g_peak) - flux_to_mag(r_peak)
            if g_late > 0 and r_late > 0:
                obj_features['g_r_color_late'] = flux_to_mag(g_late) - flux_to_mag(r_late)

            # Color change from early to late (Δ color)
            if 'g_r_color_early' in obj_features and 'g_r_color_late' in obj_features:
                obj_features['delta_g_r_color'] = obj_features['g_r_color_late'] - obj_features['g_r_color_early']

        # u-g color (UV excess detection)
        if filter_features.get('u') is not None and filter_features.get('g') is not None:
            u_peak = filter_features['u'].get('flux_peak_phase_mean', 0)
            g_peak = filter_features['g'].get('flux_peak_phase_mean', 0)
            if u_peak > 0 and g_peak > 0:
                obj_features['u_g_color_peak'] = flux_to_mag(u_peak) - flux_to_mag(g_peak)

        # 3.15: COVERAGE QUALITY METRICS
        # Good multi-band coverage is crucial for TDE identification

        # Count bands with good coverage in each phase
        n_bands_pre = sum([1 for b in ['g','r','i'] if filter_features.get(b) is not None and filter_features[b].get('n_pre_peak', 0) > 0])
        n_bands_near = sum([1 for b in ['g','r','i'] if filter_features.get(b) is not None and filter_features[b].get('n_near_peak', 0) >= 3])
        n_bands_post = sum([1 for b in ['g','r','i'] if filter_features.get(b) is not None and filter_features[b].get('n_post_peak', 0) >= 2])

        obj_features['n_bands_with_pre_peak'] = n_bands_pre
        obj_features['n_bands_with_near_peak'] = n_bands_near
        obj_features['n_bands_with_post_peak'] = n_bands_post

        # Flag for objects with good multi-band color information
        obj_features['some_color_flag'] = int(
            (n_bands_pre >= 1) and
            (n_bands_near >= 3) and
            (n_bands_post >= 2)
        )

        # U-band coverage (TDEs often have UV detections)
        has_u_near = filter_features.get('u') is not None and filter_features['u'].get('n_near_peak', 0) >= 1
        obj_features['has_u_near_peak'] = int(has_u_near)

        # 3.16: DECAY ALPHA CONSISTENCY ACROSS BANDS
        # TDEs should have consistent decay slopes across optical bands

        decay_alphas = []
        for b in ['g', 'r', 'i']:
            if filter_features.get(b) is not None:
                alpha = filter_features[b].get('decay_alpha', 0)
                if alpha > 0:  # Valid fit
                    decay_alphas.append(alpha)

        if len(decay_alphas) > 0:
            obj_features['mean_decay_alpha'] = np.mean(decay_alphas)
            obj_features['std_decay_alpha'] = np.std(decay_alphas) if len(decay_alphas) > 1 else 0

            # Flag if decay is in TDE range (α ~ 1.2-2.2)
            obj_features['decay_alpha_near_tde'] = int(
                1.2 < obj_features['mean_decay_alpha'] < 2.2
            )
        else:
            obj_features['mean_decay_alpha'] = 0
            obj_features['std_decay_alpha'] = 0
            obj_features['decay_alpha_near_tde'] = 0

        features_list.append(obj_features)

    print(f"\n✓ Feature extraction complete!")
    return pd.DataFrame(features_list)

# Extract features for train and test sets

print("\n[4/6] Extracting TRAIN features...")
train_features = create_advanced_features(train_lightcurves)
train_features = train_features.merge(
    train_log[['object_id', 'Z', 'EBV', 'target']],
    on='object_id',
    how='left'
)

print("\n[4/6] Extracting TEST features...")
test_features = create_advanced_features(test_lightcurves)
test_features = test_features.merge(
    test_log[['object_id', 'Z', 'EBV']],
    on='object_id',
    how='left'
)

# Add redshift-based features
# Redshift (Z) affects observed light curves
# - Higher Z → time dilation, dimming
# - Create interaction features

train_features['log1p_z'] = np.log1p(train_features['Z'].fillna(0))
train_features['z_squared'] = (train_features['Z'].fillna(0)) ** 2
train_features['ebv_z_int'] = train_features['EBV'].fillna(0) * train_features['Z'].fillna(0)

test_features['log1p_z'] = np.log1p(test_features['Z'].fillna(0))
test_features['z_squared'] = (test_features['Z'].fillna(0)) ** 2
test_features['ebv_z_int'] = test_features['EBV'].fillna(0) * test_features['Z'].fillna(0)

print(f"\nFeature extraction complete:")
print(f"  Train features: {train_features.shape}")
print(f"  Test features: {test_features.shape}")

# SECTION 4: PREPARE TRAINING DATA

# Separate features and target
X = train_features.drop(['object_id', 'target'], axis=1).fillna(0)
y = train_features['target'].astype(int)
X_test = test_features.drop(['object_id'], axis=1).fillna(0)

# Align feature columns between train and test
# (handle any missing features due to sparse data)
print("\n[5/6] Aligning feature columns...")
all_cols = sorted(set(X.columns) | set(X_test.columns))

for col in all_cols:
    if col not in X.columns:
        X[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

X = X[all_cols]
X_test = X_test[all_cols]

print(f"  Final feature count: {len(all_cols)}")
print(f"  Train shape: {X.shape}")
print(f"  Test shape: {X_test.shape}")

# Class imbalance information
n_tde = y.sum()
n_non_tde = (1-y).sum()
pos_weight = n_non_tde / max(n_tde, 1)

print(f"\nClass distribution:")
print(f"  TDE: {n_tde} ({100*n_tde/len(y):.2f}%)")
print(f"  Non-TDE: {n_non_tde} ({100*n_non_tde/len(y):.2f}%)")
print(f"  pos_weight: {pos_weight:.2f}")

# SECTION 5: MODEL TRAINING (ENSEMBLE OF 4 MODELS)
# Train 4 different models using 5-fold cross-validation:
#   1. XGBoost - Gradient boosting with tree-based splits
#   2. LightGBM - Fast gradient boosting with leaf-wise growth
#   3. RandomForest - Ensemble of decision trees
#   4. CatBoost - Gradient boosting optimized for categorical features
#
# Each model generates:
#   - Out-of-fold (OOF) predictions for training data
#   - Test predictions (averaged across 5 folds)

print("\n[6/6] Training ensemble models (5-fold CV)...")
print("-"*50)

# 5.1: XGBoost Configuration

xgb_params = {
    'max_depth': 7,                    # Maximum tree depth
    'learning_rate': 0.03,             # Step size shrinkage (conservative)
    'n_estimators': 1200,              # Number of boosting rounds
    'subsample': 0.75,                 # Row sampling (prevent overfitting)
    'colsample_bytree': 0.75,          # Column sampling per tree
    'min_child_weight': 5,             # Minimum sum of instance weight in child
    'gamma': 0.2,                      # Minimum loss reduction for split
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 2.0,                 # L2 regularization
    'scale_pos_weight': pos_weight,    # Balance class weights
    'random_state': 42,
    'n_jobs': -1                       # Use all CPU cores
}

# 5.2: LightGBM Configuration

lgb_params = {
    'objective': 'binary',             # Binary classification
    'learning_rate': 0.02,             # Slower learning (more conservative)
    'n_estimators': 1500,              # More trees to compensate
    'num_leaves': 48,                  # Maximum leaves per tree
    'max_depth': 8,                    # Maximum tree depth
    'subsample': 0.75,                 # Row sampling
    'colsample_bytree': 0.75,          # Column sampling
    'min_child_samples': 25,           # Minimum samples per leaf
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.5,                 # L2 regularization
    'random_state': 42,
    'is_unbalance': True,              # Handle class imbalance
    'verbose': -1                      # Suppress training logs
}

# 5.3: RandomForest Configuration

rf_params = {
    'n_estimators': 700,               # Number of trees in forest
    'max_depth': 18,                   # Maximum tree depth (deeper than boosting)
    'min_samples_split': 3,            # Minimum samples to split node
    'class_weight': 'balanced',        # Auto-balance classes
    'n_jobs': -1,
    'random_state': 42
}

# 5.4: CatBoost Configuration

cat_params = {
    'iterations': 1200,                # Number of boosting rounds
    'learning_rate': 0.03,
    'depth': 7,                        # Tree depth
    'loss_function': 'Logloss',        # Binary log loss
    'scale_pos_weight': pos_weight,    # Balance classes
    'random_seed': 42,
    'verbose': False                   # Suppress logs
}

# 5.5: 5-Fold Cross-Validation Training


# Initialize stratified K-fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Storage for out-of-fold predictions (for ensemble optimization)
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_rf = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Storage for test predictions (will average across folds)
test_xgb_folds = []
test_lgb_folds = []
test_rf_folds = []
test_cat_folds = []

# Train each fold
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/5")
    print(f"{'='*60}")

    # Split data
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    print(f"  Train: {len(X_tr)} samples ({y_tr.sum()} TDEs)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} TDEs)")

    # ------------------------------------------------------------------------
    # Train XGBoost
    # ------------------------------------------------------------------------
    print("  [1/4] Training XGBoost...")
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_tr, y_tr)

    # Predict probabilities (column 1 = probability of TDE)
    oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
    test_xgb_folds.append(m_xgb.predict_proba(X_test)[:, 1])

    # ------------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------------
    print("  [2/4] Training LightGBM...")
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_tr, y_tr)

    oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
    test_lgb_folds.append(m_lgb.predict_proba(X_test)[:, 1])

    # ------------------------------------------------------------------------
    # Train RandomForest
    # ------------------------------------------------------------------------
    print("  [3/4] Training RandomForest...")
    m_rf = RandomForestClassifier(**rf_params)
    m_rf.fit(X_tr, y_tr)

    oof_rf[val_idx] = m_rf.predict_proba(X_val)[:, 1]
    test_rf_folds.append(m_rf.predict_proba(X_test)[:, 1])

    # ------------------------------------------------------------------------
    # Train CatBoost
    # ------------------------------------------------------------------------
    print("  [4/4] Training CatBoost...")
    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
    test_cat_folds.append(m_cat.predict_proba(X_test)[:, 1])

# 5.6: Average Test Predictions Across Folds

test_xgb = np.mean(test_xgb_folds, axis=0)
test_lgb = np.mean(test_lgb_folds, axis=0)
test_rf = np.mean(test_rf_folds, axis=0)
test_cat = np.mean(test_cat_folds, axis=0)

print("\n" + "-"*50)
print("5-FOLD CROSS-VALIDATION COMPLETE")
print("-"*50)

# SECTION 6: ENSEMBLE OPTIMIZATION
# Find optimal weighted combination of 4 models
# Grid search over weight combinations + threshold

print("\n" + "-"*50)
print("OPTIMIZING WEIGHTED ENSEMBLE")
print("-"*50)

# Define weight combinations to try (wx, wl, wr, wc)
# Constraints: wx + wl + wr + wc = 1.0
weights_to_try = [
    (0.3, 0.5, 0.0, 0.2),   # Main combination (RF weight = 0)
    (0.3, 0.4, 0.1, 0.2),
    (0.25, 0.45, 0.1, 0.2),
    (0.25, 0.4, 0.15, 0.2),
    (0.2, 0.5, 0.1, 0.2),
    (0.25, 0.4, 0.1, 0.25),
    (0.2, 0.4, 0.2, 0.2),
]

best_f1 = 0.0
best_w = None
best_thr = 0.10

# Grid search
for wx, wl, wr, wc in weights_to_try:
    # Create weighted ensemble on OOF predictions
    oof_ens = wx*oof_xgb + wl*oof_lgb + wr*oof_rf + wc*oof_cat

    # Find best threshold for this weight combination
    for thr in np.arange(0.05, 0.30, 0.002):  # Scan from 5% to 30%
        pred = (oof_ens >= thr).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_w = (wx, wl, wr, wc)
            best_thr = thr

# Extract best weights
wx, wl, wr, wc = best_w

print(f"\n{'-'*50}")
print("BEST ENSEMBLE CONFIGURATION")
print(f"{'-'*50}")
print(f"OOF F1 Score:  {best_f1:.4f}")
print(f"Weights:       XGB={wx:.2f}, LGBM={wl:.2f}, RF={wr:.2f}, CAT={wc:.2f}")
print(f"Threshold:     {best_thr:.3f}")
print(f"{'-'*50}")

# Note on RandomForest weight
if wr == 0:
    print("\nNote: RandomForest has weight=0 in final ensemble")
    print("      However, it was trained and contributed to diversity during grid search")

# ============================================================================
# SECTION 7: GENERATE FINAL SUBMISSION
# ============================================================================

print("\n" + "="*70)
print("GENERATING SUBMISSION FILE")
print("="*70)

# Apply best weights to test predictions
test_ens = wx*test_xgb + wl*test_lgb + wr*test_rf + wc*test_cat

# Create submission file
submission = pd.DataFrame({
    'object_id': test_features['object_id'],
    'is_tde': (test_ens >= best_thr).astype(int)
})
submission.to_csv('submission_v3_final.csv', index=False)

# Also save probabilities for threshold tuning on leaderboard
test_probs = pd.DataFrame({
    'object_id': test_features['object_id'],
    'ensemble_prob': test_ens
})
test_probs.to_csv('test_probs_v3_final.csv', index=False)

# Summary statistics
n_tde_pred = submission['is_tde'].sum()
pct_tde = 100 * n_tde_pred / len(submission)

print(f"\n{'='*70}")
print("SUBMISSION SUMMARY")
print(f"{'='*70}")
print(f" submission_v3_final.csv")
print(f"   Total objects:     {len(submission):,}")
print(f"   Predicted TDEs:    {n_tde_pred} ({pct_tde:.2f}%)")
print(f"\n test_probs_v3_final.csv")
print(f"   (For threshold tuning on public LB)")
print(f"{'='*70}")

# FINAL RESULTS SUMMARY


print("\n" + "="*70)
print("V3 PIPELINE COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\n📊 Model Performance (OOF):")
# Calculate individual model F1 scores
def get_best_f1(oof_probs, y_true):
    best_f1 = 0
    for thr in np.arange(0.05, 0.35, 0.005):
        f1 = f1_score(y_true, (oof_probs >= thr).astype(int))
        best_f1 = max(best_f1, f1)
    return best_f1

f1_xgb = get_best_f1(oof_xgb, y)
f1_lgb = get_best_f1(oof_lgb, y)
f1_rf = get_best_f1(oof_rf, y)
f1_cat = get_best_f1(oof_cat, y)

print(f"   XGBoost:     {f1_xgb:.4f}")
print(f"   LightGBM:    {f1_lgb:.4f}")
print(f"   RandomForest:{f1_rf:.4f}")
print(f"   CatBoost:    {f1_cat:.4f}")
print(f"   Ensemble:    {best_f1:.4f} (BEST)")

print(f"\nOutput Files:")
print(f"   1. submission_v3_final.csv")
print(f"   2. test_probs_v3_final.csv")

import pandas as pd
import numpy as np

probs = pd.read_csv('test_probs_v3_final.csv')

# Thực hiện quét mịn ==> 0.217 là optimal
optimal_thresholds = [0.217]

for thr in optimal_thresholds:
    submission = pd.DataFrame({
        'object_id': probs['object_id'],
        'is_tde': (probs['ensemble_prob'] >= thr).astype(int)
    })

    n_tde = submission['is_tde'].sum()
    filename = f'submission_v3_thr{int(thr*1000)}.csv'
    submission.to_csv(filename, index=False)

    print(f"\n {filename}")
    print(f"   Threshold: {thr:.3f}")
    print(f"   TDEs: {n_tde} ({100*n_tde/len(probs):.2f}%)")

