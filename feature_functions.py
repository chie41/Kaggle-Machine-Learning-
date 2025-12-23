import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import itertools

# ============================================================================
# STEP 2: EXTINCTION CORRECTION
# ============================================================================

def apply_extinction_correction(flux, ebv, filter_name):
    R = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}
    A_lambda = R.get(filter_name, 0) * ebv
    corrected_flux = flux * 10**(0.4 * A_lambda)
    return corrected_flux

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

    # Sort by time
    sort_idx = np.argsort(t)
    t, y, yerr = t[sort_idx], y[sort_idx], yerr[sort_idx]

    # GP kernel: same as MALLORN
    kernel = 1.0 * RBF(length_scale=np.median(np.diff(t)) + 1e-3) + WhiteKernel(noise_level=np.mean(yerr)**2)

    try:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=yerr**2, normalize_y=True)
        gp.fit(t.reshape(-1, 1), y)

        # Uniform grid
        t_grid = np.arange(t.min(), t.max(), dt)

        y_mean, y_std = gp.predict(t_grid.reshape(-1, 1), return_std=True)
        
        # Derivative via finite differences
        dy = np.gradient(y_mean, dt)

        return t_grid, y_mean, y_std, dy
    except:
        return None

# ============================================================================
# STEP 3: ENHANCED FEATURE ENGINEERING (ROUND 2)
# ============================================================================

def extract_advanced_lightcurve_features(lc_group):
    """Extract comprehensive features including Round 2 improvements"""
    features = {}

    time = lc_group['Time (MJD)'].values
    flux = lc_group['Flux_corrected'].values
    flux_err = lc_group['Flux_err'].values

    if len(flux) < 3:
        return pd.Series(features)
    
    # ==========================================================
    # 1) Gaussian Process smoothing + uniform resampling (NEW)
    # ==========================================================
    gp_output = gp_smooth_and_resample(time, flux, flux_err, dt=1.0)
    if gp_output is not None:
        t_grid, f_gp, f_gp_std, f_gp_grad = gp_output

        # GP smoothed stats
        features['gp_flux_mean'] = np.mean(f_gp)
        features['gp_flux_std'] = np.std(f_gp)
        features['gp_flux_max'] = np.max(f_gp)
        features['gp_flux_min'] = np.min(f_gp)

        # GP peak features
        gp_peak_idx = np.argmax(f_gp)
        features['gp_peak_flux'] = f_gp[gp_peak_idx]
        features['gp_peak_time'] = t_grid[gp_peak_idx]

        # GP shape features
        features['gp_rise_slope'] = np.max(f_gp_grad[:gp_peak_idx]) if gp_peak_idx > 2 else 0
        features['gp_decline_slope'] = np.min(f_gp_grad[gp_peak_idx:]) if gp_peak_idx < len(f_gp_grad)-2 else 0
        features['gp_smoothness'] = np.mean(np.abs(f_gp_grad))

        # GP variance features
        features['gp_mean_var'] = np.mean(f_gp_std)
        features['gp_var_slope'] = (f_gp_std[-1] - f_gp_std[0]) / (t_grid[-1] - t_grid[0] + 1e-10)

    # ===== BASIC STATISTICS =====
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_median'] = np.median(flux)
    features['flux_max'] = np.max(flux)
    features['flux_min'] = np.min(flux)
    features['flux_range'] = features['flux_max'] - features['flux_min']

    # ===== HIGHER-ORDER MOMENTS =====
    features['flux_skew'] = pd.Series(flux).skew()
    features['flux_kurtosis'] = pd.Series(flux).kurtosis()

    # ===== EXTENDED PERCENTILES (NEW!) =====
    features['flux_05th'] = np.percentile(flux, 5)
    features['flux_10th'] = np.percentile(flux, 10)
    features['flux_25th'] = np.percentile(flux, 25)
    features['flux_75th'] = np.percentile(flux, 75)
    features['flux_90th'] = np.percentile(flux, 90)
    features['flux_95th'] = np.percentile(flux, 95)
    features['flux_iqr'] = features['flux_75th'] - features['flux_25th']
    features['flux_iqr_ratio'] = features['flux_iqr'] / (features['flux_median'] + 1e-10)

    # ===== TEMPORAL FEATURES =====
    features['duration'] = time.max() - time.min()
    features['n_observations'] = len(lc_group)

    if len(time) > 1:
        time_diffs = np.diff(np.sort(time))
        features['mean_time_gap'] = np.mean(time_diffs)
        features['median_time_gap'] = np.median(time_diffs)
        features['std_time_gap'] = np.std(time_diffs)
        features['max_time_gap'] = np.max(time_diffs)
        features['min_time_gap'] = np.min(time_diffs)
    else:
        features['mean_time_gap'] = 0
        features['median_time_gap'] = 0
        features['std_time_gap'] = 0
        features['max_time_gap'] = 0
        features['min_time_gap'] = 0

    # ===== PEAK FEATURES =====
    peak_idx = np.argmax(flux)
    features['peak_flux'] = flux[peak_idx]
    features['time_to_peak'] = time[peak_idx] - time.min() if len(time) > 0 else 0
    features['time_after_peak'] = time.max() - time[peak_idx] if len(time) > 0 else 0
    features['peak_position_ratio'] = features['time_to_peak'] / (features['duration'] + 1e-10)

    # ===== RISE AND DECLINE =====
    if peak_idx > 0 and peak_idx < len(flux) - 1:
        features['rise_rate'] = features['peak_flux'] / (features['time_to_peak'] + 1e-10)
        features['decline_rate'] = features['peak_flux'] / (features['time_after_peak'] + 1e-10)
        features['rise_decline_ratio'] = features['rise_rate'] / (features['decline_rate'] + 1e-10)

        # Symmetry (NEW!)
        features['rise_decline_symmetry'] = abs(features['time_to_peak'] - features['time_after_peak']) / (features['duration'] + 1e-10)
    else:
        features['rise_rate'] = 0
        features['decline_rate'] = 0
        features['rise_decline_ratio'] = 0
        features['rise_decline_symmetry'] = 0

    # ===== VARIABILITY METRICS =====
    features['amplitude'] = (features['flux_max'] - features['flux_min']) / (features['flux_max'] + features['flux_min'] + 1e-10)
    features['beyond_1std'] = np.sum(np.abs(flux - features['flux_mean']) > features['flux_std']) / len(flux)
    features['beyond_2std'] = np.sum(np.abs(flux - features['flux_mean']) > 2*features['flux_std']) / len(flux)

    # ===== FLUX ABOVE BASELINE =====
    features['flux_above_median_frac'] = np.sum(flux > features['flux_median']) / len(flux)
    features['flux_positive_frac'] = np.sum(flux > 0) / len(flux)
    features['flux_above_mean_frac'] = np.sum(flux > features['flux_mean']) / len(flux)

    # ===== WEIGHTED STATISTICS =====
    weights = 1 / (flux_err**2 + 1e-10)
    features['flux_weighted_mean'] = np.average(flux, weights=weights)
    features['flux_weighted_std'] = np.sqrt(np.average((flux - features['flux_weighted_mean'])**2, weights=weights))

    # ===== CHI-SQUARED VARIABILITY =====
    features['chi2_per_dof'] = np.sum((flux - features['flux_mean'])**2 / (flux_err**2 + 1e-10)) / (len(flux) - 1 + 1e-10)

    # ===== FLUX CHANGES =====
    if len(flux) > 1:
        flux_changes = np.diff(flux)
        features['max_flux_change'] = np.max(np.abs(flux_changes))
        features['mean_abs_flux_change'] = np.mean(np.abs(flux_changes))
        features['std_flux_change'] = np.std(flux_changes)

        # Consecutive increases/decreases (NEW!)
        increasing = flux_changes > 0
        decreasing = flux_changes < 0
        features['max_consecutive_inc'] = max([sum(1 for _ in g) for k, g in __import__('itertools').groupby(increasing) if k], default=0)
        features['max_consecutive_dec'] = max([sum(1 for _ in g) for k, g in __import__('itertools').groupby(decreasing) if k], default=0)
    else:
        features['max_flux_change'] = 0
        features['mean_abs_flux_change'] = 0
        features['std_flux_change'] = 0
        features['max_consecutive_inc'] = 0
        features['max_consecutive_dec'] = 0

    # ===== TIME-WEIGHTED FEATURES (NEW!) =====
    # Split into early vs late observations
    time_split = np.median(time)
    early_mask = time <= time_split
    late_mask = time > time_split

    if early_mask.sum() > 0 and late_mask.sum() > 0:
        features['flux_early_mean'] = np.mean(flux[early_mask])
        features['flux_late_mean'] = np.mean(flux[late_mask])
        features['flux_evolution'] = features['flux_late_mean'] - features['flux_early_mean']
        features['flux_evolution_ratio'] = features['flux_late_mean'] / (features['flux_early_mean'] + 1e-10)
    else:
        features['flux_early_mean'] = features['flux_mean']
        features['flux_late_mean'] = features['flux_mean']
        features['flux_evolution'] = 0
        features['flux_evolution_ratio'] = 1

    # ===== ROBUST STATISTICS (NEW!) =====
    features['mad'] = np.median(np.abs(flux - features['flux_median']))  # Median Absolute Deviation
    features['flux_rms'] = np.sqrt(np.mean(flux**2))

    return pd.Series(features)


def create_advanced_features(lightcurves_df):
    """Create advanced features for each object with multi-band combinations"""
    features_list = []

    unique_objects = lightcurves_df['object_id'].unique()
    print(f"\nExtracting advanced features from {len(unique_objects)} objects...")

    for idx, obj_id in enumerate(unique_objects):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(unique_objects)} objects...")

        obj_lc = lightcurves_df[lightcurves_df['object_id'] == obj_id]
        obj_features = {'object_id': obj_id}

        # Store per-filter features
        filter_features = {}

        # Extract features per filter
        for filter_name in ['u', 'g', 'r', 'i', 'z', 'y']:
            filter_lc = obj_lc[obj_lc['Filter'] == filter_name]

            if len(filter_lc) >= 3:
                features = extract_advanced_lightcurve_features(filter_lc)
                filter_features[filter_name] = features

                for feat_name, feat_val in features.items():
                    obj_features[f'{filter_name}_{feat_name}'] = feat_val
            else:
                filter_features[filter_name] = None

        # ===== MULTI-BAND COLOR FEATURES =====
        if 'g' in filter_features and 'r' in filter_features and filter_features['g'] is not None and filter_features['r'] is not None:
            obj_features['color_g_minus_r'] = filter_features['g']['flux_mean'] - filter_features['r']['flux_mean']
            obj_features['color_g_minus_r_peak'] = filter_features['g']['peak_flux'] - filter_features['r']['peak_flux']
            obj_features['color_g_minus_r_median'] = filter_features['g']['flux_median'] - filter_features['r']['flux_median']

            # Flux ratios (NEW!)
            obj_features['flux_ratio_g_over_r'] = filter_features['g']['flux_mean'] / (filter_features['r']['flux_mean'] + 1e-10)
            obj_features['flux_ratio_g_over_r_peak'] = filter_features['g']['peak_flux'] / (filter_features['r']['peak_flux'] + 1e-10)

        if 'r' in filter_features and 'i' in filter_features and filter_features['r'] is not None and filter_features['i'] is not None:
            obj_features['color_r_minus_i'] = filter_features['r']['flux_mean'] - filter_features['i']['flux_mean']
            obj_features['color_r_minus_i_peak'] = filter_features['r']['peak_flux'] - filter_features['i']['peak_flux']
            obj_features['color_r_minus_i_median'] = filter_features['r']['flux_median'] - filter_features['i']['flux_median']
            obj_features['flux_ratio_r_over_i'] = filter_features['r']['flux_mean'] / (filter_features['i']['flux_mean'] + 1e-10)

        if 'g' in filter_features and 'i' in filter_features and filter_features['g'] is not None and filter_features['i'] is not None:
            obj_features['color_g_minus_i'] = filter_features['g']['flux_mean'] - filter_features['i']['flux_mean']
            obj_features['flux_ratio_g_over_i'] = filter_features['g']['flux_mean'] / (filter_features['i']['flux_mean'] + 1e-10)

        if 'i' in filter_features and 'z' in filter_features and filter_features['i'] is not None and filter_features['z'] is not None:
            obj_features['color_i_minus_z'] = filter_features['i']['flux_mean'] - filter_features['z']['flux_mean']
            obj_features['flux_ratio_i_over_z'] = filter_features['i']['flux_mean'] / (filter_features['z']['flux_mean'] + 1e-10)

        # ===== PEAK TIMING DIFFERENCES =====
        if 'r' in filter_features and 'i' in filter_features and filter_features['r'] is not None and filter_features['i'] is not None:
            obj_features['peak_time_diff_r_i'] = filter_features['r']['time_to_peak'] - filter_features['i']['time_to_peak']

        if 'g' in filter_features and 'r' in filter_features and filter_features['g'] is not None and filter_features['r'] is not None:
            obj_features['peak_time_diff_g_r'] = filter_features['g']['time_to_peak'] - filter_features['r']['time_to_peak']

        if 'g' in filter_features and 'i' in filter_features and filter_features['g'] is not None and filter_features['i'] is not None:
            obj_features['peak_time_diff_g_i'] = filter_features['g']['time_to_peak'] - filter_features['i']['time_to_peak']

        # ===== VARIABILITY RATIOS ACROSS BANDS (NEW!) =====
        if 'r' in filter_features and 'i' in filter_features and filter_features['r'] is not None and filter_features['i'] is not None:
            obj_features['amplitude_ratio_r_i'] = filter_features['r']['amplitude'] / (filter_features['i']['amplitude'] + 1e-10)
            obj_features['std_ratio_r_i'] = filter_features['r']['flux_std'] / (filter_features['i']['flux_std'] + 1e-10)

        features_list.append(obj_features)

    return pd.DataFrame(features_list)