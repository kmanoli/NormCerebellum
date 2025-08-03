#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr

# Configure plotting style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8
})

#####################################
### DATA IMPORT AND PREPROCESSING ###
#####################################

# Set directory
data_dir = '/project/normative_cerebellum'

# Import data
data = pd.read_csv(os.path.join(data_dir, 'behavioral_outcomes/zscores_behaviors.csv'))

behavs = data.iloc[:, 1:11]  # Behavioral scores
parcels = data.iloc[:, 14:] # Parcel normative z-scores

behav_names = behavs.columns.tolist()
parcel_names = parcels.columns.tolist()

age = data['age']
sex = data['sex']

def residualize_data(X, confounds):
    """
    Regress out the confounds from each column of X.
    X: (n_samples, n_features)
    confounds: (n_samples, n_confounds)
    Returns: residualized X (same shape as X)
    """
    # Add an intercept to confounds
    conf = np.column_stack([np.ones(confounds.shape[0]), confounds])
    beta, _, _, _ = np.linalg.lstsq(conf, X, rcond=None)
    X_resid = X - conf @ beta
    return X_resid


# Residualize behaviors for age and sex (to match parcels)
confounds_beh = data[["age", "sex"]].values
behavs_r = residualize_data(behavs, confounds_beh)

# Scale data 
scalerX = StandardScaler()
scalerY = StandardScaler()
X_scaled = scalerX.fit_transform(parcels)
Y_scaled = scalerY.fit_transform(behavs_r)

###############################################
### PLS COMPONENTS AND COVARIANCE EXPLAINED ###
###############################################

print("\n--- Calculating PLS singular values and covariance explained ---")

# Maximum number of components to analyze
max_components = min(10, min(X_scaled.shape[1], Y_scaled.shape[1]))

# Fit the PLS model with max_components
pls = PLSRegression(n_components=max_components, scale=False)
pls.fit(X_scaled, Y_scaled)

# Get X and Y scores
X_scores = pls.transform(X_scaled)
Y_pred = pls.predict(X_scaled)

# Calculate singular values (equivalent to square root of eigenvalues of X'Y Y'X)
singvals = np.zeros(max_components)
for i in range(max_components):
    # Calculate covariance between X and Y scores
    cov_matrix = np.cov(X_scores[:, i], Y_pred[:, i])
    # Scale by sqrt(n-1) to get singval equivalent
    singvals[i] = cov_matrix[0, 1] * np.sqrt(X_scaled.shape[0] - 1)

# Calculate percentage of covariance explained by each component
cv = singvals**2 / np.sum(singvals**2) * 100
print("Covariance explained by each component (%):", cv)

# Permutation test for null distribution
print("\n--- Running permutation tests to generate null distribution ---")

# Number of permutations
n_permutations = 10000

# Store permuted singular values
perm_singvals = np.zeros((max_components, n_permutations))

# Run permutation tests
for perm in range(n_permutations):
    # Shuffle Y
    Y_perm = np.random.permutation(Y_scaled)
    
    # Fit PLS on permuted data
    pls_perm = PLSRegression(n_components=max_components, scale=False)
    pls_perm.fit(X_scaled, Y_perm)
    
    # Get permuted scores
    X_perm_scores = pls_perm.transform(X_scaled)
    Y_perm_pred = pls_perm.predict(X_scaled)
    
    # Calculate permuted singular values
    for i in range(max_components):
        cov_matrix = np.cov(X_perm_scores[:, i], Y_perm_pred[:, i])
        perm_singvals[i, perm] = cov_matrix[0, 1] * np.sqrt(X_scaled.shape[0] - 1)
    
    if (perm + 1) % 100 == 0:
        print(f"Completed {perm + 1} permutations")

# Calculate percentage of covariance explained in the permuted models
cv_perms = np.zeros_like(perm_singvals)
for p in range(n_permutations):
    cv_perms[:, p] = perm_singvals[:, p]**2 / np.sum(perm_singvals[:, p]**2) * 100

# Calculate p-values for each component
p_values = np.zeros(max_components)
for i in range(max_components):
    p_values[i] = (1 + np.sum(perm_singvals[i, :] > singvals[i])) / (1 + n_permutations)

# Print summary
print("\n--- Summary of Components ---")
for i in range(max_components):
    print(f"Component {i+1}: {cv[i]:.2f}% covariance explained, p-value = {p_values[i]:.3f}")

# Identify significant components
sig_components = np.where(p_values < 0.05)[0] + 1
print(f"\nSignificant components (p < 0.05): {sig_components}")

# Plot 
print("\n--- Creating visualization ---")

plt.figure(figsize=(7, 6))

box_positions = range(max_components)

boxplots = plt.boxplot(cv_perms.T, positions=box_positions, widths=0.6, 
                       patch_artist=True, 
                       boxprops=dict(facecolor='white', color='gray', linewidth=0.8),
                       whiskerprops=dict(color='gray', linewidth=0.8),
                       capprops=dict(color='gray', linewidth=0.8),
                       medianprops=dict(color='gray', linewidth=1.2),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=3,
                                      markeredgecolor='gray', alpha=0.5))

plt.scatter(box_positions, cv, s=60, c='#F9A03F', label='Effect size', 
            edgecolor='black', linewidth=0.5, zorder=10)
plt.ylim(0, 100) 
plt.xlabel('Latent variable', fontsize=10)
plt.ylabel('% covariance explained', fontsize=10)
plt.tick_params(axis='both', which='both', direction='out', length=0)
plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='y', labelsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.2)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F9A03F', 
              markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Effect size'),
    Rectangle((0, 0), 1, 1, fc='white', ec='gray', linewidth=0.8, label='Spin null')
]
plt.legend(handles=legend_elements, frameon=False, fontsize=9, loc='upper right')
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(data_dir,  'behavioral_outcomes', 'pls_covariance_explained.png'), bbox_inches='tight', dpi=300)

##############################################################
### PLS CROSS-VALIDATION AND PERMUTATION SIGNIFICANCE TEST ###
##############################################################

# Start with unscaled data
X = parcels
Y = behavs_r

# Set up cross-validation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store results
train_correlations = []
test_correlations = []

# Perform cross-validation with normalization within each fold
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    print(f"\n=== Outer Fold {fold} ===")
    
    # Split the data
    X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
    X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]
    
    # Scale within the fold using only training data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    # Fit the model on scaled training data
    model = PLSRegression(n_components=1, scale=False)  # scale=False since we manually scaled
    model.fit(X_train_scaled, Y_train_scaled)
    
    # Transform both train and test data
    X_train_latent = model.transform(X_train_scaled)
    Y_train_latent = Y_train_scaled @ model.y_weights_  
    
    X_test_latent = model.transform(X_test_scaled)
    Y_test_latent = Y_test_scaled @ model.y_weights_
    
    # Compute correlation for training data (between latent variables)
    train_corr, _ = pearsonr(X_train_latent[:, 0], Y_train_latent[:, 0])
    train_correlations.append(train_corr)
    
    # Compute correlation for test data
    test_corr, _ = pearsonr(X_test_latent[:, 0], Y_test_latent[:, 0])
    test_correlations.append(test_corr)
    
    print(f"Fold {fold} Train Canonical Correlation: {train_corr:.3f}")
    print(f"Fold {fold} Test Canonical Correlation: {test_corr:.3f}")

# Calculate statistics
train_correlations = np.array(train_correlations)
test_correlations = np.array(test_correlations)
avg_train_corr = np.mean(np.abs(train_correlations))
avg_test_corr = np.mean(np.abs(test_correlations))
print(f"\nAverage absolute train CV canonical correlation: {avg_train_corr:.3f}")
print(f"Average absolute test CV canonical correlation: {avg_test_corr:.3f}")

# Perform permutation test with standardization on the full dataset
n_permutations = 10000  

# Normalize the full dataset for the final model
scaler_X_full = StandardScaler()
scaler_Y_full = StandardScaler()
X_scaled_full = scaler_X_full.fit_transform(X)
Y_scaled_full = scaler_Y_full.fit_transform(Y)

# Fit final model on normalized full data
final_model = PLSRegression(n_components=1, scale=False)
final_model.fit(X_scaled_full, Y_scaled_full)

# Get latent scores
X_latent_full = final_model.transform(X_scaled_full)
Y_latent_full = Y_scaled_full @ final_model.y_weights_
real_corr, _ = pearsonr(X_latent_full[:, 0], Y_latent_full[:, 0])

# Store permuted correlations
permuted_correlations = np.zeros(n_permutations)

# Run the permutation test
for i in range(n_permutations):
    # Shuffle Y (breaks relationship with X)
    Y_permuted = np.random.permutation(Y_scaled_full)
    
    # Fit PLS on shuffled data
    permuted_model = PLSRegression(n_components=1, scale=False)
    permuted_model.fit(X_scaled_full, Y_permuted)
    
    # Transform X and get the Y scores
    X_perm_latent = permuted_model.transform(X_scaled_full)
    Y_perm_latent = Y_permuted @ permuted_model.y_weights_
    
    # Compute canonical correlation on shuffled data
    permuted_correlations[i], _ = pearsonr(X_perm_latent[:, 0], Y_perm_latent[:, 0])
    
    # Progress update every 100 permutations
    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1}/{n_permutations} permutations...")

# Compute the p-value (proportion of permuted correlations >= real correlation)
p_value = np.mean(np.abs(permuted_correlations) >= np.abs(real_corr))
print(f"Real Canonical Correlation: {real_corr:.3f}")
print(f"Permutation Test p-value: {p_value:.5f}")

# Print summary statistics 
print("\nPLS Cross-Validation Performance Summary:")
print("─" * 50)
print(f"Training correlation (mean ± SD): {np.mean(train_correlations):.3f} ± {np.std(train_correlations):.3f}")
print(f"Test correlation (mean ± SD): {np.mean(test_correlations):.3f} ± {np.std(test_correlations):.3f}")
print(f"Null correlation (mean ± SD): {np.mean(permuted_correlations):.3f} ± {np.std(permuted_correlations):.3f}")
print("─" * 50)
print(f"Permutation test p-value: {p_value:.5f}")
if p_value < 0.05:
    print(f"Result: Significant (p < 0.05)")
else:
    print(f"Result: Not significant (p ≥ 0.05)")

# Plot cross-validation
plt.figure(figsize=(7, 2))

# Prepare data for boxplots
boxplot_data = [train_correlations, test_correlations, permuted_correlations]
boxplot_labels = ['Train', 'Test', 'Null']

plt.figure(figsize=(7, 2))
bp = plt.boxplot(
    boxplot_data, 
    labels=boxplot_labels, 
    patch_artist=True,
    widths=0.4, 
    showfliers=False,
    medianprops={'color': 'black', 'linewidth': 1.2},
    whiskerprops={'color': 'black', 'linewidth': 0.8},
    capprops={'color': 'black', 'linewidth': 0.8},
    vert=False  # This makes the boxplot horizontal
)

for i, box in enumerate(bp['boxes']):
    box.set(facecolor='white', edgecolor='#333333', linewidth=0.8, alpha=0.9)

plt.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
plt.grid(axis='x', linestyle='--', alpha=0.2)
plt.xlabel("Score correlation (Pearson's r)", fontsize=9)
plt.ylabel("", fontsize=9)  # Empty ylabel
plt.tick_params(axis='both', which='both', direction='out', length=0)
plt.tick_params(axis='y', labelsize=9)
plt.tick_params(axis='x', labelsize=8)
all_values = np.concatenate(boxplot_data)
x_min = min(np.min(all_values) * 1.1, -0.1)
x_max = max(np.max(all_values) * 1.1, 0.8)
plt.xlim(x_min, x_max)
plt.tight_layout(pad=0.5)

# Save figure 
plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'pls_cv_boxplots.png', dpi=300, bbox_inches='tight'))

######################################################
### EXTRACT EMPIRICAL PLS CORRELATION AND LOADINGS ###
######################################################

final_model = PLSRegression(n_components=1, scale=False)  # scale=False since we already scaled
final_model.fit(X_scaled, Y_scaled)

# Convert to numpy arrays 
X_scaled_np = X_scaled.values if hasattr(X_scaled, 'values') else X_scaled
Y_scaled_np = Y_scaled.values if hasattr(Y_scaled, 'values') else Y_scaled

X_pls = final_model.transform(X_scaled_np)
Y_pls = Y_scaled_np @ final_model.y_weights_

real_corr, _ = pearsonr(X_pls[:, 0], Y_pls[:, 0])

# Retrieve loadings 
brain_loadings = final_model.x_loadings_  
behavioral_loadings = final_model.y_loadings_ 

print("Final in-sample canonical correlation (full data):", real_corr)

# Plot behavior loadings
# Get loadings for the first component and sort by absolute values
loading_values = behavioral_loadings[:, 0]
abs_loadings = np.abs(loading_values)
sorted_indices = np.argsort(abs_loadings)  # Sort in ascending order of absolute values
sorted_loadings = loading_values[sorted_indices]
sorted_vars = np.array(behav_names)[sorted_indices]


plt.figure(figsize=(8, 10))
ax = plt.gca()

for i, (pos, loading) in enumerate(zip(range(len(sorted_vars)), sorted_loadings)):
    if loading >= 0:
        color = 'darkorange'
    else:
        color = 'royalblue'
    ax.barh(pos, loading, color=color, height=0.7)

ax.set_yticks(range(len(sorted_vars)))
ax.set_yticklabels(sorted_vars, fontsize=10)
ax.set_xlabel("Loadings", fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which='both', left=False)
ax.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'behavior_loadings.png'), dpi=300,  bbox_inches='tight')

# Plot brain loadings
# Get loadings for the first component and sort by absolute values
loading_values = brain_loadings[:, 0]
abs_loadings = np.abs(loading_values)
sorted_indices = np.argsort(abs_loadings)  # Sort in ascending order of absolute values
sorted_loadings = loading_values[sorted_indices]
sorted_vars = np.array(parcel_names)[sorted_indices]

plt.figure(figsize=(8, 10))
ax = plt.gca()

for i, (pos, loading) in enumerate(zip(range(len(sorted_vars)), sorted_loadings)):
    if loading >= 0:
        color = 'darkviolet'
    else:
        color = 'black'
    ax.barh(pos, loading, color=color, height=0.7)

ax.set_yticks(range(len(sorted_vars)))
ax.set_yticklabels(sorted_vars, fontsize=10)
ax.set_xlabel("Loadings", fontsize=12)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)  # Changed to gray for better visibility against black bars
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which='both', left=False)
ax.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'brain_loadings.png'), dpi=300,  bbox_inches='tight')

# Plot empirical relationship between brain and behavior scores
plt.figure(figsize=(6, 5))

ax = plt.gca()
plt.scatter(X_pls[:, 0], Y_pls[:, 0], 
          alpha=0.8, 
          s=60,  
          color='darkviolet',  
          edgecolors='white',
          linewidth=0.5)

# Add regression line
z = np.polyfit(X_pls[:, 0], Y_pls[:, 0], 1)
p = np.poly1d(z)
x_range = np.linspace(X_pls[:, 0].min(), X_pls[:, 0].max(), 100)
plt.plot(x_range, p(x_range), 
        color='darkorange', 
        linewidth=4,
        linestyle='-',
        alpha=0.8)

# Add 95% CI
X_with_intercept = sm.add_constant(X_pls[:, 0])
model = sm.OLS(Y_pls[:, 0], X_with_intercept).fit()
X_range_with_intercept = sm.add_constant(x_range)
predictions = model.get_prediction(X_range_with_intercept)
ci = predictions.conf_int()

plt.fill_between(x_range, ci[:, 0], ci[:, 1], 
               color='darkorange', alpha=0.2, label='95% CI')
plt.xlabel('Brain Scores (LV1)', fontsize=11, fontweight='normal')
plt.ylabel('Behavior Scores (LV1)', fontsize=11, fontweight='normal')
plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='both', direction='out', length=4, width=0.8)
plt.tick_params(axis='both', which='major', labelsize=9)
x_padding = (X_pls[:, 0].max() - X_pls[:, 0].min()) * 0.05
y_padding = (Y_pls[:, 0].max() - Y_pls[:, 0].min()) * 0.05
plt.xlim(X_pls[:, 0].min() - x_padding, X_pls[:, 0].max() + x_padding)
plt.ylim(Y_pls[:, 0].min() - y_padding, Y_pls[:, 0].max() + y_padding)
plt.tight_layout()

plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'pls_empirical_correlation.png'), 
            dpi=300, bbox_inches='tight')

#########################
### PLS BOOTSTRAPPING ###
#########################

print("\n--- Starting Bootstrap Analysis ---")

# Number of bootstrap samples
n_bootstrap = 10000

# Arrays to store bootstrap results
bootstrap_brain_loadings = np.zeros((brain_loadings.shape[0], n_bootstrap))
bootstrap_behavioral_loadings = np.zeros((behavioral_loadings.shape[0], n_bootstrap))
bootstrap_correlations = np.zeros(n_bootstrap)

# Run the bootstrap
for i in range(n_bootstrap):
    # Resample with replacement
    boot_indices = np.random.choice(X_scaled.shape[0], size=X_scaled.shape[0], replace=True)
    X_boot = X_scaled[boot_indices]
    Y_boot = Y_scaled[boot_indices]
    
    # Fit PLS on bootstrap sample
    boot_model = PLSRegression(n_components=1, scale=False)
    boot_model.fit(X_boot, Y_boot)
    
    # Store the loadings
    bootstrap_behavioral_loadings[:, i] = boot_model.y_weights_[:, 0]
    bootstrap_brain_loadings[:, i] = boot_model.x_weights_[:, 0]
    
    # Store the correlation 
    X_boot_pls = boot_model.transform(X_boot)
    Y_boot_pls = boot_model.predict(X_boot)
    boot_corr, _ = pearsonr(X_boot_pls[:, 0], Y_boot_pls[:, 0])
    bootstrap_correlations[i] = boot_corr
    
    # Show progress
    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1} bootstrap samples")

# Calculate bootstrap statistics
brain_loading_means = np.mean(bootstrap_brain_loadings, axis=1)
brain_loading_std = np.std(bootstrap_brain_loadings, axis=1)
brain_bootstrap_ratios = brain_loading_means / brain_loading_std

behavioral_loading_means = np.mean(bootstrap_behavioral_loadings, axis=1)
behavioral_loading_std = np.std(bootstrap_behavioral_loadings, axis=1)
behavioral_bootstrap_ratios = behavioral_loading_means / behavioral_loading_std

# Calculate 95% confidence intervals
brain_ci_lower = np.percentile(bootstrap_brain_loadings, 2.5, axis=1)
brain_ci_upper = np.percentile(bootstrap_brain_loadings, 97.5, axis=1)

behavioral_ci_lower = np.percentile(bootstrap_behavioral_loadings, 2.5, axis=1)
behavioral_ci_upper = np.percentile(bootstrap_behavioral_loadings, 97.5, axis=1)

# Print bootstrap correlation statistics
print(f"\nBootstrap Correlation Statistics:")
print(f"Mean: {np.mean(bootstrap_correlations):.3f}")
print(f"Standard Deviation: {np.std(bootstrap_correlations):.3f}")
print(f"95% CI: [{np.percentile(bootstrap_correlations, 2.5):.3f}, {np.percentile(bootstrap_correlations, 97.5):.3f}]")

# Determine significance using bootstrap ratios (BSR)
# Threshold: |BSR| > 2.57 (≈p<0.01)

bsr_threshold = 2.57  

# For behavioral variables
behavioral_significant_bsr = np.abs(behavioral_bootstrap_ratios) > bsr_threshold

# For brain variables  
brain_significant_bsr = np.abs(brain_bootstrap_ratios) > bsr_threshold

# Print results
print("\n--- Behavioral Variables Significance ---")
for i, var in enumerate(behav_names):
    ci_sig = behavioral_ci_lower[i] * behavioral_ci_upper[i] > 0
    bsr_sig = behavioral_significant_bsr[i]
    print(f"{var}: BSR={behavioral_bootstrap_ratios[i]:.2f}, "
          f"CI=[{behavioral_ci_lower[i]:.3f}, {behavioral_ci_upper[i]:.3f}], "
          f"CI sig: {ci_sig}, BSR sig: {bsr_sig}")

print("\n--- Brain Variables Significance (top 10 by |BSR|) ---")
# Sort by absolute BSR for display
brain_bsr_order = np.argsort(np.abs(brain_bootstrap_ratios))[::-1]
for idx in brain_bsr_order[:10]:
    ci_sig = brain_ci_lower[idx] * brain_ci_upper[idx] > 0
    bsr_sig = brain_significant_bsr[idx]
    print(f"{parcel_names[idx]}: BSR={brain_bootstrap_ratios[idx]:.2f}, "
          f"CI=[{brain_ci_lower[idx]:.3f}, {brain_ci_upper[idx]:.3f}], "
          f"CI sig: {ci_sig}, BSR sig: {bsr_sig}")
    
# Plot behavioral bootstrap
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(behav_names))

# Plot the bars
plt.bar(y_pos, behavioral_loading_means, color='darkorange', alpha=0.7)

# Add error bars for 95% CIs
plt.errorbar(y_pos, behavioral_loading_means, 
             yerr=[behavioral_loading_means - behavioral_ci_lower, 
                   behavioral_ci_upper - behavioral_loading_means],
             fmt='none', ecolor='black', capsize=3)

# Highlight significant variables (CI doesn't cross 0)
significant_vars = (behavioral_ci_lower * behavioral_ci_upper > 0)
for i, sig in enumerate(significant_vars):
    if sig:
        plt.bar(y_pos[i], behavioral_loading_means[i], color='red', alpha=0.5)

plt.xticks(y_pos, behav_names, rotation=45, ha='right')
plt.xlabel('Behavioral Variables')
plt.ylabel('Mean Bootstrap Loading')
plt.title('Behavioral Loadings with 95% Bootstrap Confidence Intervals')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'behavior_bootstrap.png'), dpi=300,  bbox_inches='tight')

# Plot brain bootstrap
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(parcel_names))

# Plot the bars
plt.bar(y_pos, brain_loading_means, color='blue', alpha=0.7)

# Add error bars for 95% CIs
plt.errorbar(y_pos, brain_loading_means, 
             yerr=[brain_loading_means - brain_ci_lower, 
                   brain_ci_upper - brain_loading_means],
             fmt='none', ecolor='black', capsize=3)

# Highlight significant variables (CI doesn't cross 0)
significant_vars = (brain_ci_lower * brain_ci_upper > 0)
for i, sig in enumerate(significant_vars):
    if sig:
        plt.bar(y_pos[i], brain_loading_means[i], color='red', alpha=0.5)

plt.xticks(y_pos, parcel_names, rotation=45, ha='right')
plt.xlabel('Brain Variables')
plt.ylabel('Mean Bootstrap Loading')
plt.title('Brain Loadings with 95% Bootstrap Confidence Intervals')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', 'brain_bootstrap.png'), dpi=300,  bbox_inches='tight')
