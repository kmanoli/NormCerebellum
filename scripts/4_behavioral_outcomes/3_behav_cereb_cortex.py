#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

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

# Set up data
dat = pd.read_csv(os.path.join(data_dir, 'behavioral_outcomes/cortex_cereb_behaviors.csv'))
cerebellar_data = dat.iloc[:, 14:46]
cortical_data = dat.iloc[:, 46:]
behaviors = dat.iloc[:, 1:11]

# Save behavioral and parcel names 
cortical_names = cortical_data.columns.tolist()
cerebellar_names = cerebellar_data.columns.tolist()
behavior_names = behaviors.columns.tolist()

###############################################
### REGULARIZED REGRESSION MODEL COMPARISON ###
###############################################

models = {
    'Ridge': RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=10),
    'Lasso': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=10, max_iter=10000),
    'ElasticNet': ElasticNetCV(
        alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
        cv=10,
        max_iter=10000
    )
}

# Set up repeated cross-validation
n_splits  = 10
n_repeats = 10  

cv_outer = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=42
)

# Initialize results dictionaries
results = {
    model_name: {
        'Cortex': {behav: [] for behav in behavior_names},
        'Cerebellum': {behav: [] for behav in behavior_names},
        'Combined': {behav: [] for behav in behavior_names}
    } for model_name in models.keys()
}

# For each behavior and model type, collect R² values across folds
print(f"Comparing regularization methods with repeated nested CV "
      f"({n_repeats}×{n_splits}-fold outer CV, inner CV=10)...")

for behavior in behavior_names:
    print(f"  Processing behavior: {behavior}")
    y = behaviors[behavior].values
    
    # Use unscaled data for cross-validation splitting
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(cortical_data), start=1):        
        
        # Scale data within each fold
        scaler_cortex = StandardScaler()
        scaler_cerebellum = StandardScaler()
        
        # Fit scalers on training data only
        cortical_train_scaled = scaler_cortex.fit_transform(cortical_data.iloc[train_idx])
        cerebellar_train_scaled = scaler_cerebellum.fit_transform(cerebellar_data.iloc[train_idx])
        
        # Transform test data using training scalers
        cortical_test_scaled = scaler_cortex.transform(cortical_data.iloc[test_idx])
        cerebellar_test_scaled = scaler_cerebellum.transform(cerebellar_data.iloc[test_idx])
        
        # For each model type
        for model_name, model_class in models.items():
            # Cortex only
            model_cortex = model_class.__class__(**model_class.get_params())
            model_cortex.fit(cortical_train_scaled, y[train_idx])
            y_pred_cortex = model_cortex.predict(cortical_test_scaled)
            r2_cortex = r2_score(y[test_idx], y_pred_cortex)
            results[model_name]['Cortex'][behavior].append(r2_cortex)
            
            # Cerebellum only
            model_cereb = model_class.__class__(**model_class.get_params())
            model_cereb.fit(cerebellar_train_scaled, y[train_idx])
            y_pred_cereb = model_cereb.predict(cerebellar_test_scaled)
            r2_cereb = r2_score(y[test_idx], y_pred_cereb)
            results[model_name]['Cerebellum'][behavior].append(r2_cereb)
            
            # Combined - concatenate scaled features
            X_combined_train = np.concatenate([cortical_train_scaled, cerebellar_train_scaled], axis=1)
            X_combined_test = np.concatenate([cortical_test_scaled, cerebellar_test_scaled], axis=1)
            
            model_combined = model_class.__class__(**model_class.get_params())
            model_combined.fit(X_combined_train, y[train_idx])
            y_pred_combined = model_combined.predict(X_combined_test)
            r2_combined = r2_score(y[test_idx], y_pred_combined)
            results[model_name]['Combined'][behavior].append(r2_combined)

#########################################
### CALCULATE MEAN AND STANDARD ERROR ###
#########################################

# Calculate mean R² and standard error for each model, feature set, and behavior separately
mean_r2 = {
    model_name: {
        data_type: {
            behav: np.mean(results[model_name][data_type][behav]) 
            for behav in behavior_names
        } for data_type in ['Cortex', 'Cerebellum', 'Combined']
    } for model_name in models.keys()
}

sem_r2 = {
    model_name: {
        data_type: {
            behav: np.std(results[model_name][data_type][behav], ddof=1) / np.sqrt(len(results[model_name][data_type][behav]))
            for behav in behavior_names
        } for data_type in ['Cortex', 'Cerebellum', 'Combined']
    } for model_name in models.keys()
}

# Create dataframe of mean R² for each model, feature set, and behavior
df_list = []
for model_name in models.keys():
    for data_type in ['Cortex', 'Cerebellum', 'Combined']:
        for behav in behavior_names:
            df_list.append({
                'Model': model_name,
                'Data Type': data_type,
                'Behavior': behav,
                'R²': mean_r2[model_name][data_type][behav],
                'SEM': sem_r2[model_name][data_type][behav]
            })

df_r2 = pd.DataFrame(df_list)

# Compute mean R² across feature sets for each model × behavior
mean_r2_across_sets = {
    model_name: {
        behav: np.mean([
            mean_r2[model_name]['Cortex'][behav],
            mean_r2[model_name]['Cerebellum'][behav],
            mean_r2[model_name]['Combined'][behav]
        ])
        for behav in behavior_names
    }
    for model_name in models.keys()
}

# Compute SEM across feature sets
sem_r2_across_sets = {
    model_name: {
        behav: np.std([
            mean_r2[model_name]['Cortex'][behav],
            mean_r2[model_name]['Cerebellum'][behav],
            mean_r2[model_name]['Combined'][behav]
        ], ddof=1) / np.sqrt(3)  # divide by sqrt(3 feature sets)
        for behav in behavior_names
    }
    for model_name in models.keys()
}

# Create dataframe of mean R² across feature sets
df_avg_list = []
for model_name in models.keys():
    for behav in behavior_names:
        df_avg_list.append({
            'Model': model_name,
            'Behavior': behav,
            'R²': mean_r2_across_sets[model_name][behav],
            'SEM': sem_r2_across_sets[model_name][behav]
        })

df_avg_r2 = pd.DataFrame(df_avg_list)

##############################
#### MODEL COMPARISON PLOT ###
##############################

plt.figure(figsize=(7, 6))

# Filter for socio-linguistic behaviors
selected_behaviors = ['lang_compr', 'reading', 'srs']
df_filtered_avg = df_avg_r2[df_avg_r2['Behavior'].isin(selected_behaviors)]

# Define color palette
palette = ['silver', 'dimgrey', '#C4226F'] # Ridge, Lasso, ElasticNet

# Plot mean R² (averaged across feature sets)
ax = plt.gca()
g = sns.barplot(
    x='Behavior', 
    y='R²', 
    hue='Model', 
    data=df_filtered_avg,
    order=selected_behaviors,
    palette=palette,
    alpha=0.95
)

# Add error bars 
for i, model_name in enumerate(['Ridge', 'Lasso', 'ElasticNet']):
    for j, behavior in enumerate(selected_behaviors):
        # Get the subset of data for this model and behavior
        subset = df_filtered_avg[(df_filtered_avg['Model'] == model_name) & 
                             (df_filtered_avg['Behavior'] == behavior)]
        
        # Skip if no data available
        if len(subset) == 0:
            continue
            
        # Get the position within each behavior group
        bar_positions = np.arange(len(selected_behaviors))
        # Width of a group of bars
        width = 0.8
        # Width of an individual bar
        bar_width = width / 3
        # Calculate the position for this specific bar
        pos = bar_positions[j] + (i - 1) * bar_width
        
        # Get height and error for this bar
        height = subset['R²'].values[0]
        error = subset['SEM'].values[0]
        
        # Add error bar (thinner and cleaner)
        plt.errorbar(x=pos, y=height, yerr=error, fmt='none', color='#555555', capsize=2, linewidth=0.8)

plt.legend(title=None, frameon=False)
plt.xlabel('Behavior', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(data_dir,  'behavioral_outcomes', 'reg_model_comparison.png'), bbox_inches='tight', dpi=300)

#####################################
### ELASTICNET PLOT FOR BEHAVIORS ###
#####################################

# Filter data for ElasticNet only
df_elasticnet = df_r2[df_r2['Model'] == 'ElasticNet']

# Filter for only socio-linguistic behaviors
df_elasticnet_filtered = df_elasticnet[df_elasticnet['Behavior'].isin(selected_behaviors)]

plt.figure(figsize=(7, 7))

# Assign colors
color_dict = {
    'Cortex': '#F68E5B',    # Orange/yellow for Cortex
    'Cerebellum': '#C4226F', # Pink for Cerebellum
    'Combined': '#5D2F8E'    # Purple for Combined
}

# Create vertical bar plot manually for better control of error bars
ax = plt.gca()

# Define positions for each behavior
behavior_positions = {behavior: i for i, behavior in enumerate(selected_behaviors)}

# Define offsets for each data type within a behavior group
offsets = {'Cortex': -0.25, 'Cerebellum': 0, 'Combined': 0.25}
bar_width = 0.2

# Track min and max values for axis limits
min_val = 0
max_val = 0

# Manually plot each bar and its error bar
for data_type in ['Cortex', 'Cerebellum', 'Combined']:
    color = color_dict[data_type]
    for behavior in selected_behaviors:
        # Get data for this combination
        subset = df_elasticnet_filtered[(df_elasticnet_filtered['Data Type'] == data_type) &
                                (df_elasticnet_filtered['Behavior'] == behavior)]
        
        if len(subset) == 0:
            continue
            
        # Get R² and SEM
        r2_value = subset['R²'].values[0]
        sem_value = subset['SEM'].values[0]
        
        # Track min/max values for axis limits
        min_val = min(min_val, r2_value - sem_value)
        max_val = max(max_val, r2_value + sem_value)
        
        # Calculate position
        x_pos = behavior_positions[behavior] + offsets[data_type]
        
        # Draw the bar
        ax.bar(x_pos, r2_value, width=bar_width, color=color, alpha=0.95, 
               label=data_type if behavior == selected_behaviors[0] else "")
        
        # Draw the error bar
        ax.errorbar(x_pos, r2_value, yerr=sem_value, fmt='none', color='#555555', 
                   capsize=3, linewidth=1.0, ecolor='#555555')

plt.xticks([behavior_positions[b] for b in selected_behaviors], selected_behaviors, fontsize=11, rotation=0)
plt.xlabel('', fontsize=12)  # Remove x-axis label since behavior names are clear
plt.ylabel('R²', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.yticks(fontsize=10)
padding = (max_val - min_val) * 0.1
plt.ylim(min_val - padding, max_val + padding)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
plt.legend(title=None, frameon=False, loc='upper left')
plt.tight_layout()

plt.savefig(os.path.join(data_dir,  'behavioral_outcomes', 'enet_behaviors.png'), bbox_inches='tight', dpi=300)

################################
### STATISTICAL COMPARISONS ###
################################

data_types = ['Cortex', 'Cerebellum', 'Combined']
behavior_names_test = ['srs', 'lang_compr', 'reading']  
model_name = 'ElasticNet'

comparison_results = []

for behavior in behavior_names_test:
    for i in range(len(data_types)):
        for j in range(i + 1, len(data_types)):
            dt_a = data_types[i]
            dt_b = data_types[j]
            
            # R² values across all outer folds × repeats
            r2_a = results[model_name][dt_a][behavior]
            r2_b = results[model_name][dt_b][behavior]

            # Sanity check: Must be same length and aligned by fold
            assert r2_a.shape == r2_b.shape, "Mismatch in fold counts for Wilcoxon"

            # Wilcoxon signed-rank test
            stat, p = wilcoxon(r2_a, r2_b, zero_method='wilcox', mode='approx') # 'approx' because N will be large with repeated CV

            # Count non-zero diffs
            n = np.sum(np.abs(np.array(r2_a) - np.array(r2_b)) > 0)

            comparison_results.append({
                'Behavior': behavior,
                'Data Type A': dt_a,
                'Data Type B': dt_b,
                'Mean R2 A': np.mean(r2_a),
                'SD R2 A': np.std(r2_a, ddof=1),
                'Median R2 A': np.median(r2_a),
                'Mean R2 B': np.mean(r2_b),
                'SD R2 B': np.std(r2_b, ddof=1),
                'Median R2 B': np.median(r2_b),
                'Wilcoxon W': stat,
                'Wilcoxon p-value': p,
                'n (non-zero diffs)': n,
                'A better?': np.median(r2_a) > np.median(r2_b),
                'Significant (p < 0.05)': p < 0.05
            })

df_dt_comparisons = pd.DataFrame(comparison_results)

for behavior in behavior_names_test:
    print(f"\n=== Data type comparisons for {behavior.upper()} ===")
    df_bh = df_dt_comparisons[df_dt_comparisons['Behavior'] == behavior]
    if df_bh.empty:
        continue
    for _, row in df_bh.iterrows():
        dt1 = row['Data Type A']
        dt2 = row['Data Type B']
        W = row['Wilcoxon W']
        p = row['Wilcoxon p-value']
        n = row['n (non-zero diffs)']

        mean1 = row['Mean R2 A']
        sd1 = row['SD R2 A']
        mean2 = row['Mean R2 B']
        sd2 = row['SD R2 B']

        # Format p-value
        p_str = "< .001" if p < 0.001 else f"= {p:.3f}"

        # Who performed better
        better = dt1 if row['A better?'] else dt2
        worse = dt2 if row['A better?'] else dt1

        print(
            f"  {better} outperformed {worse}: "
            f"Wilcoxon signed-rank test, W = {W:.2f}, n = {n}, p {p_str}. "
            f"{dt1}: M = {mean1:.3f}, SD = {sd1:.3f}; {dt2}: M = {mean2:.3f}, SD = {sd2:.3f}"
        )
