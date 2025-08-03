#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import fdrcorrection
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
import time

# Configure plotting style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8
})


# Set directory
data_dir = '/project/normative_cerebellum'

# Load and prepare data
dat = pd.read_csv(os.path.join(data_dir, 'cerebral_associations/cortex_cereb_zscores.csv'))
cerebellar_data = dat.iloc[:, 0:32]
cortical_data = dat.iloc[:, 33:]

# Save behavioral and parcel names 
cortical_names = cortical_data.columns.tolist()
cereb_names = cerebellar_data.columns.tolist()

###############################################
### REGULARIZED REGRESSION MODEL COMPARISON ###
###############################################

# Setup for nested cross-validation
alphas = np.logspace(-3, 3, 7)  # Range of regularization strengths
cv_folds = 10
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Set up models with internal CV for hyperparameter tuning
models = {
    'Ridge': RidgeCV(alphas=alphas, cv=10),
    'Lasso': LassoCV(alphas=alphas, cv=10, max_iter=10000),
    'ElasticNet': ElasticNetCV(
        alphas=alphas,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],  # Test multiple l1_ratios
        cv=10, 
        max_iter=10000
    )
}

# Initialize results storage
results = {
    model_name: {cortical_name: [] for cortical_name in cortical_names}
    for model_name in models.keys()
}

# Process each cortical parcel
print("Comparing regularization methods using nested cross-validation...")
for i, cortical_name in enumerate(cortical_names):
    print(f"Processing cortical parcel: {cortical_name} ({i+1}/{len(cortical_names)})")
    
    # Get raw data for this parcel
    y_raw = cortical_data.iloc[:, i].values
    X_raw = cerebellar_data.values
    
    # For each model type
    for model_name, model_class in models.items():
        cv_scores = []
        
        # Outer CV loop for unbiased evaluation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw)):
            # Split data before scaling
            X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
            y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]
            
            # Scale within each fold
            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train_raw)
            X_test = X_scaler.transform(X_test_raw)
            
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()
            
            # Create a fresh instance of the model
            # This will perform internal CV on the training data to select best alpha
            model = model_class.__class__(**model_class.get_params())
            
            # Fit model (internal CV happens here)
            model.fit(X_train, y_train)
            
            # Evaluate on test fold
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            cv_scores.append(r2)
        
        # Store mean CV score for this model and parcel
        results[model_name][cortical_name] = cv_scores

# Calculate average performance across all parcels for each model
mean_r2_per_model = {}
std_r2_per_model = {}

for model_name in models.keys():
    # Flatten all CV scores across all parcels
    all_scores = []
    for cortical_name in cortical_names:
        all_scores.extend(results[model_name][cortical_name])
    
    # Calculate mean and std across all parcels
    mean_r2_per_model[model_name] = np.mean(all_scores)
    std_r2_per_model[model_name] = np.std(all_scores)

# Determine best model
best_model = max(mean_r2_per_model.items(), key=lambda x: x[1])[0]
best_r2 = mean_r2_per_model[best_model]

print(f"\nBest model: {best_model} with average R²={best_r2:.4f}")

# Create model comparison plot
plt.figure(figsize=(8, 6))

model_names = list(models.keys())
r2_values = [mean_r2_per_model[model] for model in model_names]
r2_stds = [std_r2_per_model[model] for model in model_names]

# Define colors to match original
colors = {'Ridge': 'salmon', 'Lasso': 'silver', 'ElasticNet': 'dimgrey'}
bar_colors = [colors[model] for model in model_names]

bars = plt.bar(
    model_names, 
    r2_values, 
    color=bar_colors,
    width=0.6,
    yerr=r2_stds, 
    capsize=5,
    error_kw={'ecolor': '#555555', 'linewidth': 1, 'capthick': 1, 'alpha': 0.8}
)

plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.xlabel('Model', fontsize=12, labelpad=10)
plt.ylabel('R² Score', fontsize=12, labelpad=10)
plt.title('Model Performance Comparison (Nested CV)', fontsize=14, pad=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.ylim(bottom=-0.02)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cerebral_associations/model_comparison.png'), dpi=300, bbox_inches='tight')

#######################################################
### CEREBRAL CORTEX ASSOCIATIONS WITH WINNING MODEL ###
#######################################################

# First fit the winning model on all data to get final weights and optimal alpha
print(f"\nFitting {best_model} on all data to extract weights and optimal parameters...")

# Standardize all data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(cerebellar_data.values)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(cortical_data.values)

# Fit the best model type on all data to get the optimal alpha
if best_model == 'Ridge':
    final_model = RidgeCV(alphas=alphas, cv=10)
elif best_model == 'Lasso':
    final_model = LassoCV(alphas=alphas, cv=10, max_iter=10000)
else:  # ElasticNet
    final_model = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=10, max_iter=10000)

# Fit on first cortical parcel to get the selected alpha (and l1_ratio for ElasticNet)
final_model.fit(X_scaled, y_scaled[:, 0])
best_alpha = final_model.alpha_

if best_model == 'ElasticNet':
    best_l1_ratio = final_model.l1_ratio_
    print(f"Optimal alpha selected by {best_model}: {best_alpha:.4f}")
    print(f"Optimal l1_ratio selected by {best_model}: {best_l1_ratio:.2f}")
else:
    best_l1_ratio = None
    print(f"Optimal alpha selected by {best_model}: {best_alpha:.4f}")

# Save model selection results
if best_model == 'ElasticNet':
    results_text = f"""Model Selection Results (Nested Cross-Validation)
================================================

Winning Model: {best_model}
Optimal Alpha: {best_alpha:.4f}
Optimal L1 Ratio: {best_l1_ratio:.2f}
Cross-validated R²: {best_r2:.4f} ± {std_r2_per_model[best_model]:.4f}

All Models Comparison:
- Ridge: R²={mean_r2_per_model['Ridge']:.4f} ± {std_r2_per_model['Ridge']:.4f}
- Lasso: R²={mean_r2_per_model['Lasso']:.4f} ± {std_r2_per_model['Lasso']:.4f}  
- ElasticNet: R²={mean_r2_per_model['ElasticNet']:.4f} ± {std_r2_per_model['ElasticNet']:.4f}
"""
else:
    results_text = f"""Model Selection Results (Nested Cross-Validation)
================================================

Winning Model: {best_model}
Optimal Alpha: {best_alpha:.4f}
Cross-validated R²: {best_r2:.4f} ± {std_r2_per_model[best_model]:.4f}

All Models Comparison:
- Ridge: R²={mean_r2_per_model['Ridge']:.4f} ± {std_r2_per_model['Ridge']:.4f}
- Lasso: R²={mean_r2_per_model['Lasso']:.4f} ± {std_r2_per_model['Lasso']:.4f}  
- ElasticNet: R²={mean_r2_per_model['ElasticNet']:.4f} ± {std_r2_per_model['ElasticNet']:.4f}
"""

with open(os.path.join(data_dir, 'cerebral_associations/model_selection_results.txt'), 'w') as f:
    f.write(results_text)

# Now apply the best model with the selected alpha to all cortical parcels
results_df = pd.DataFrame(index=cortical_names, columns=['R2', 'Top10_Regions', 'Top10_Coefficients'])
weights_matrix = np.zeros((len(cortical_names), len(cereb_names)))

for i, cortical_name in enumerate(cortical_names):
    if i % 10 == 0:
        print(f"Analyzing parcel: {cortical_name} ({i+1}/{len(cortical_names)})")
    
    y = y_scaled[:, i]
    X = X_scaled
    
    # Use the best model with the optimal alpha (and l1_ratio for ElasticNet)
    if best_model == 'Ridge':
        model = Ridge(alpha=best_alpha, max_iter=10000, random_state=42)
    elif best_model == 'Lasso':
        model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
    else:  # ElasticNet
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
    
    # Fit on all data
    model.fit(X, y)
    weights_matrix[i, :] = model.coef_
    
    # Calculate R² on the full dataset
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Get coefficients and identify top regions
    coefs = model.coef_
    importance = np.abs(coefs)
    sorted_idx = np.argsort(importance)[::-1]
    
    # Get all cerebellar regions sorted by importance
    all_regions = [cereb_names[j] for j in sorted_idx]
    all_coefs = [coefs[j] for j in sorted_idx]

    # Store results
    results_df.loc[cortical_name, 'R2'] = r2
    results_df.loc[cortical_name, 'All_Regions'] = str(all_regions)
    results_df.loc[cortical_name, 'All_Coefficients'] = str(all_coefs)

# Sort the results by R²
results_df = pd.DataFrame(index=cortical_names, columns=['R2', 'All_Regions', 'All_Coefficients'])

# Plot bar chart of R² values for all parcels
plt.figure(figsize=(14, 10))
plt.bar(range(len(results_df)), results_df['R2'].astype(float).values)
plt.xticks(range(len(results_df)), results_df.index, rotation=90)
plt.xlabel('Cortical Parcels')
plt.ylabel('R² Value')
plt.title(f'Prediction Performance for All Cortical Parcels (Using {best_model})')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cerebral_associations/all_parcels_performance.png'), dpi=300, bbox_inches='tight')

# Save results
results_df.to_csv(os.path.join(data_dir, 'cerebral_associations/all_parcels_performance.csv'))

# Create a dataframe for the weights matrix
weights_df = pd.DataFrame(weights_matrix, index=cortical_names, columns=cereb_names)
weights_df.to_csv(os.path.join(data_dir, 'cerebral_associations/cerebral_assoc.csv'))

# Plot heatmap of weights
plt.figure(figsize=(18, 12))
im = plt.imshow(weights_matrix, cmap='bwr', aspect='auto', vmin=-0.3, vmax=0.3)
plt.colorbar(im, label='Weight Coefficient')
plt.xlabel('Cerebellar Regions (Predictors)')
plt.ylabel('Cortical Regions (Targets)')
plt.title(f'Weights Between Cerebellar and Cortical Regions ({best_model} Model)')
plt.xticks(range(len(cereb_names)), cereb_names, rotation=90)
plt.yticks(range(len(cortical_names)), cortical_names)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cerebral_associations/cerebral_assoc_uncorr.png'), dpi=300, bbox_inches='tight')

########################################
### PERMUTATION SIGNIFICANCE TESTING ###
########################################

# Permutation testing for significance
print("\nStarting permutation testing...")
p_values_matrix = np.zeros((len(cortical_names), len(cereb_names)))

total_pairs = len(cortical_names) * len(cereb_names)
processed_pairs = 0
start_time = time.time()
n_permutations = 1000

print(f"Testing {total_pairs} cortical-cerebellar pairs")
print(f"Using {n_permutations} permutations per pair")

for i, cortical_name in enumerate(cortical_names):
    y = y_scaled[:, i]
    X = X_scaled
    
    # Initialize model with best alpha (and l1_ratio for ElasticNet)
    if best_model == 'Ridge':
        model = Ridge(alpha=best_alpha, max_iter=10000, random_state=42)
    elif best_model == 'Lasso':
        model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
    else:
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
    
    model.fit(X, y)
    orig_coefs = model.coef_
    
    for j, cereb_name in enumerate(cereb_names):
        processed_pairs += 1
        if processed_pairs % 100 == 0:
            elapsed_time = time.time() - start_time
            percent_done = (processed_pairs / total_pairs) * 100
            estimated_remaining = (elapsed_time / percent_done * 100 - elapsed_time) if percent_done > 0 else 0
            print(f"  Progress: {percent_done:.1f}% complete, Est. remaining: {estimated_remaining/60:.1f} min")
        
        orig_coef = orig_coefs[j]
        perm_coefs = []
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            model.fit(X, y_perm)
            perm_coefs.append(model.coef_[j])
        
        # Calculate p-value
        if orig_coef >= 0:
            p_value = np.mean(np.array(perm_coefs) >= orig_coef)
        else:
            p_value = np.mean(np.array(perm_coefs) <= orig_coef)
        
        p_values_matrix[i, j] = p_value

print(f"\nPermutation testing completed in {(time.time() - start_time)/60:.1f} minutes")

# Create dataframe and save p-values
p_values_df = pd.DataFrame(p_values_matrix, index=cortical_names, columns=cereb_names)
p_values_df.to_csv(os.path.join(data_dir, 'cerebral_associations/cerebral_assoc_pvalues_uncorr.csv'))

# Apply FDR correction
alpha_threshold = 0.05
flat_pvals = p_values_matrix.flatten()
rejected, p_corrected = fdrcorrection(flat_pvals, alpha=alpha_threshold, method='indep')
p_values_matrix_fdr = p_corrected.reshape(p_values_matrix.shape)
significance_mask_fdr = p_values_matrix_fdr > alpha_threshold

p_values_fdr_df = pd.DataFrame(p_values_matrix_fdr, index=cortical_names, columns=cereb_names)

##################################
### FDR-CORRECTED ASSOCIATIONS ###
##################################

# Create masked heatmap of significant weights
plt.figure(figsize=(10, 10))
sns.heatmap(
    data=weights_df, 
    mask=significance_mask_fdr, 
    cmap='coolwarm', 
    vmin=-0.2, vmax=0.2,
    center=0, 
    annot=False, 
    cbar_kws={'label': 'Weight Coefficient (FDR-corrected, p<.05)'}
)
plt.xlabel('Cerebellar Regions (Predictors)')
plt.ylabel('Cortical Regions (Targets)')
plt.title(f'Significant Associations Between Cerebellar and Cortical Regions\n({best_model} Model, FDR-corrected p<{alpha_threshold})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cerebral_associations/cerebral_assoc_fdr.png'), dpi=300, bbox_inches='tight')

# Average hemispheres for cleaner visualization
cortical_names_no_prefix = []
region_mapping = {}

for name in cortical_names:
    if name.startswith('rh_') or name.startswith('lh_'):
        region_name = name[3:]  
        if region_name not in region_mapping:
            region_mapping[region_name] = []
        region_mapping[region_name].append(name)
        
        if region_name not in cortical_names_no_prefix:
            cortical_names_no_prefix.append(region_name)
    else:
        cortical_names_no_prefix.append(name)
        region_mapping[name] = [name]

# Create averaged data
averaged_weights_data = {}
averaged_p_values_data = {}

for region in cortical_names_no_prefix:
    original_regions = region_mapping[region]
    
    if len(original_regions) > 1:
        rows_to_average = weights_df.loc[original_regions]
        averaged_weights_data[region] = rows_to_average.mean(axis=0)
        
        p_rows_to_use = p_values_fdr_df.loc[original_regions]
        averaged_p_values_data[region] = p_rows_to_use.min(axis=0)
    else:
        averaged_weights_data[region] = weights_df.loc[original_regions[0]]
        averaged_p_values_data[region] = p_values_fdr_df.loc[original_regions[0]]

averaged_weights_df = pd.DataFrame(averaged_weights_data).T
averaged_p_values_fdr = pd.DataFrame(averaged_p_values_data).T
averaged_significance_mask = averaged_p_values_fdr > alpha_threshold

# Create display labels
display_labels = {}
for region in cortical_names_no_prefix:
    if '_' in region:
        parts = region.split('_')
        abbreviated = ' '.join([p.capitalize() for p in parts])
    else:
        abbreviated = region.capitalize()
    display_labels[region] = abbreviated

# Plot averaged heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    data=averaged_weights_df, 
    mask=averaged_significance_mask, 
    cmap='coolwarm', 
    vmin=-0.2, vmax=0.2,
    center=0, 
    annot=False, 
    cbar_kws={'label': 'Weight Coefficient (FDR-corrected, p<0.05)'}
)
plt.xlabel('Cerebellar Regions (Predictors)', fontsize=12)
plt.ylabel('Cortical Regions (Targets)', fontsize=12)
plt.title(f'Significant Associations (Averaged Across Hemispheres)\n({best_model} Model, FDR-corrected p<{alpha_threshold})')
plt.xticks(rotation=70, fontsize=14)
plt.yticks(
    ticks=np.arange(len(cortical_names_no_prefix)) + 0.5,
    labels=[display_labels[region] for region in cortical_names_no_prefix],
    rotation=0, 
    fontsize=14
)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cerebral_associations/ccerebral_assoc_fdr_avg.png'), dpi=300, bbox_inches='tight')

#####################
### PLOT ON BRAIN ###
#####################

# Load the DK atlas in surface space
dk_lh = nib.load('/project/normative_cerebellum/atlases/lh.aparc.label.gii')
dk_lh_data = dk_lh.agg_data()
dk_rh = nib.load('/project/normative_cerebellum/atlases/rh.aparc.label.gii')
dk_rh_data = dk_rh.agg_data()

# Load the DK label lookup table
dk_lut = pd.read_csv(os.path.join(data_dir, 'atlas_labels/dk_labels.csv'))
print(f"DK lookup table shape: {dk_lut.shape}")

# Process the LUT - remove "ctx-" prefix from names and standardize format
dk_lut['clean_name'] = dk_lut['label'].str.replace('ctx-', '')  # Remove ctx- prefix
dk_lut['clean_name'] = dk_lut['clean_name'].str.replace('lh-', 'lh_')  # Replace lh- with lh_
dk_lut['clean_name'] = dk_lut['clean_name'].str.replace('rh-', 'rh_')  # Replace rh- with rh_

# Print sample of lookup table for debugging
print("Sample of processed lookup table:")
print(dk_lut[['roi', 'label', 'clean_name']].head(5))

# Get unique values from the surface data for debugging
lh_unique_labels = np.unique(dk_lh_data)
rh_unique_labels = np.unique(dk_rh_data)
print(f"Unique values in LH surface data: {lh_unique_labels[:10]}...")
print(f"Unique values in RH surface data: {rh_unique_labels[:10]}...")

# Create a direct mapping DataFrame
mapping_df = pd.DataFrame({
    'cortical_name': cortical_names,
    'lut_label': None,    # Will store LUT indices (from lookup table)
    'img_label': None     # Will store actual image indices (0-35)
})

# Populate lut_label column by matching names with the lookup table
for i, name in enumerate(cortical_names):
    # Find exact match after removing prefix
    match = dk_lut[dk_lut['clean_name'] == name]
    if not match.empty:
        mapping_df.loc[i, 'lut_label'] = match['roi'].values[0]
    else:
        # Try alternative formats if exact match fails
        if name.startswith('lh_'):
            alt_name = 'lh-' + name[3:]
            match = dk_lut[dk_lut['clean_name'] == alt_name]
        elif name.startswith('rh_'):
            alt_name = 'rh-' + name[3:]
            match = dk_lut[dk_lut['clean_name'] == alt_name]
        
        if not match.empty:
            mapping_df.loc[i, 'lut_label'] = match['roi'].values[0]

# Fill the img_label column based on a direct mapping between LUT labels and image labels
# For the right hemisphere, the mapping is 2000->0, 2001->1, etc.
# For the left hemisphere, the mapping is 1000->0, 1001->1, etc.
for i, row in mapping_df.iterrows():
    if pd.notna(row['lut_label']):
        name = row['cortical_name']
        lut_value = int(row['lut_label'])
        
        if name.startswith('rh_'):
            # Determine if we need to map using offset or direct lookup
            for img_idx, unique_label in enumerate(rh_unique_labels):
                if img_idx >= 2:  # Skip 0 and 1 as they don't correspond to DK parcels
                    if lut_value == 2000 + img_idx:
                        mapping_df.loc[i, 'img_label'] = unique_label
                        break
        elif name.startswith('lh_'):
            for img_idx, unique_label in enumerate(lh_unique_labels):
                if img_idx >= 2:  # Skip 0 and 1 as they don't correspond to DK parcels
                    if lut_value == 1000 + img_idx:
                        mapping_df.loc[i, 'img_label'] = unique_label
                        break

# Filter to keep only rows with valid mappings
valid_mapping_df = mapping_df.dropna(subset=['img_label']).copy()
valid_mapping_df['img_label'] = valid_mapping_df['img_label'].astype(int)

print(f"Created mapping for {len(valid_mapping_df)} out of {len(cortical_names)} cortical regions")
print("Sample of mapping dataframe:")
print(valid_mapping_df.head(5))

# Convert to dictionary for easy lookup during weight assignment
cortical_to_img_label = dict(zip(valid_mapping_df['cortical_name'], valid_mapping_df['img_label']))

# Get surfaces for visualization of all FDR weights
surfaces = fetch_fslr()
lh_surf, rh_surf = surfaces['inflated']
sulc_lh, sulc_rh = surfaces['sulc']

# In the heatmap, brain_mask_fdr=TRUE means "hide this value"
# For brain visualization, we need brain_mask=TRUE to mean "use this value"
brain_mask = ~significance_mask_fdr  # Invert the mask

# Use heatmap min and max
global_min = -0.2 
global_max = 0.2  
cmap = "coolwarm"  

# First, calculate all weight arrays for plotting
all_weight_arrays = []
for cereb_idx, cereb_name in enumerate(cereb_names):
    # Skip if no significant connections for this cerebellar region
    if not np.any(brain_mask[:, cereb_idx]):
        print(f"Skipping {cereb_name}: no significant strong connections")
        continue
    
    print(f"Processing cerebellar region: {cereb_name}")
    
    # Create weight arrays directly from the surface data
    lh_weight_data = np.zeros_like(dk_lh_data, dtype=float)
    rh_weight_data = np.zeros_like(dk_rh_data, dtype=float)
    
    # Only assign weights for cortical regions that are in the brain_mask
    for cort_idx, cort_name in enumerate(cortical_names):
        # Only process if this cortical region has a significant strong connection
        if brain_mask[cort_idx, cereb_idx]:
            weight = weights_matrix[cort_idx, cereb_idx]
            
            # Determine hemisphere and assign to the corresponding surface
            if cort_name.startswith('lh_') and cort_name in cortical_to_img_label:
                # Get the label for this cortical region
                label = cortical_to_img_label[cort_name]
                # Assign weight to all vertices with this label
                mask = dk_lh_data == label
                count = np.sum(mask)
                if count > 0:
                    lh_weight_data[mask] = weight
                else:
                    print(f"    Warning: No vertices found for {cort_name} (label {label})")
                    
            elif cort_name.startswith('rh_') and cort_name in cortical_to_img_label:
                # Get the label for this cortical region
                label = cortical_to_img_label[cort_name]
                # Assign weight to all vertices with this label
                mask = dk_rh_data == label
                count = np.sum(mask)
                if count > 0:
                    rh_weight_data[mask] = weight
                else:
                    print(f"    Warning: No vertices found for {cort_name} (label {label})")
    
    # Check if any weights were assigned
    lh_max = np.max(np.abs(lh_weight_data)) if np.any(lh_weight_data) else 0
    rh_max = np.max(np.abs(rh_weight_data)) if np.any(rh_weight_data) else 0
    print(f"    Max absolute weight value - LH: {lh_max:.4f}, RH: {rh_max:.4f}")
    
    # Store the weight arrays for this cerebellar region
    all_weight_arrays.append((cereb_name, lh_weight_data, rh_weight_data))

# Now create plots with the fixed min/max values
for cereb_name, lh_weight_data, rh_weight_data in all_weight_arrays:
    try:
        # Create plot with fixed color scale range
        print(f"Creating plot for {cereb_name} with fixed range: {global_min} to {global_max}")
        
        # Create plot
        p = Plot(lh_surf, rh_surf)
        p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r', cbar=False)
        p.add_layer({'left': lh_weight_data, 'right': rh_weight_data}, 
                   cmap=cmap, 
                   color_range=(global_min, global_max),
                   cbar_label='Weight')
        
        # Save figure
        fig = p.build()
        out_fig = os.path.join(data_dir, 'cerebral_associations', f'dk_{cereb_name}_assoc.png')
        fig.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved surface plot: {out_fig}")
        
    except Exception as e:
        print(f"  Error creating surface plot: {e}")

