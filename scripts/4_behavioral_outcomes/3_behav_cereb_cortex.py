import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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

# Set up models for comparison (hyperparameter tuning via nested ten-fold cross-validation)
models = {
    'Ridge': RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=10),
    'Lasso': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=10, max_iter=10000),
    'ElasticNet': ElasticNetCV(
        alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
        cv=10, max_iter=10000
    )
}

# Set up cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize results dictionaries
results = {
    model_name: {
        'Cortex': {behav: [] for behav in behavior_names},
        'Cerebellum': {behav: [] for behav in behavior_names},
        'Combined': {behav: [] for behav in behavior_names}
    } for model_name in models.keys()
}

# For each behavior and model type, collect R² values across folds
print("Comparing regularization methods with error bars...")

for behavior in behavior_names:
    print(f"  Processing behavior: {behavior}")
    y = behaviors[behavior].values
    
    # Use unscaled data for cross-validation splitting
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(cortical_data)):
        print(f"    Fold {fold_idx+1}/10")
        
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

# Calculate mean and standard error for each model, data type, and behavior
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

# Create dataframe for easier plotting
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

###############################
#### MODEL COMPARISON PLOT ###
##############################

plt.figure(figsize=(7, 6))

# Filter for combined models only and selected behaviors
selected_behaviors = ['lang_compr', 'reading', 'srs']
df_filtered = df_r2[(df_r2['Data Type'] == 'Combined') & 
                    (df_r2['Behavior'].isin(selected_behaviors))]

# Use the specified order for behaviors
behavior_order = selected_behaviors

# Define mixed palette: lighter greys for Ridge and Lasso, pink for ElasticNet
mixed_palette = ['silver', 'dimgrey', '#C4226F']

# Plot bars 
ax = plt.gca()
g = sns.barplot(
    x='Behavior', 
    y='R²', 
    hue='Model', 
    data=df_filtered,
    order=behavior_order,
    palette=mixed_palette,
    alpha=0.95
)

# Add error bars 
for i, model_name in enumerate(['Ridge', 'Lasso', 'ElasticNet']):
    for j, behavior in enumerate(behavior_order):
        # Get the subset of data for this model and behavior
        subset = df_filtered[(df_filtered['Model'] == model_name) & 
                             (df_filtered['Behavior'] == behavior)]
        
        # Skip if no data available
        if len(subset) == 0:
            continue
            
        # Get the position within each behavior group
        bar_positions = np.arange(len(behavior_order))
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

# Filter for only the requested behaviors
selected_behaviors = ['lang_compr', 'reading', 'srs']
df_elasticnet_filtered = df_elasticnet[df_elasticnet['Behavior'].isin(selected_behaviors)]

# Set custom behavior order
behavior_order = ['lang_compr', 'reading', 'srs']

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
behavior_positions = {behavior: i for i, behavior in enumerate(behavior_order)}

# Define offsets for each data type within a behavior group
offsets = {'Cortex': -0.25, 'Cerebellum': 0, 'Combined': 0.25}
bar_width = 0.2

# Track min and max values for axis limits
min_val = 0
max_val = 0

# Manually plot each bar and its error bar
for data_type in ['Cortex', 'Cerebellum', 'Combined']:
    color = color_dict[data_type]
    for behavior in behavior_order:
        # Get data for this combination
        subset = df_elasticnet[(df_elasticnet['Data Type'] == data_type) & 
                             (df_elasticnet['Behavior'] == behavior)]
        
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
               label=data_type if behavior == behavior_order[0] else "")
        
        # Draw the error bar
        ax.errorbar(x_pos, r2_value, yerr=sem_value, fmt='none', color='#555555', 
                   capsize=3, linewidth=1.0, ecolor='#555555')

plt.xticks([behavior_positions[b] for b in behavior_order], behavior_order, fontsize=11, rotation=0)
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

            r2_a = results[model_name][dt_a][behavior]
            r2_b = results[model_name][dt_b][behavior]

            # Wilcoxon signed-rank test
            stat, p = wilcoxon(r2_a, r2_b, zero_method='wilcox', mode='exact' if len(r2_a) <= 25 else 'approx')

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