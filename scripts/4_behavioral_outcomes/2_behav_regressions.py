import os
import nibabel as nib 
import numpy as np
import pandas as pd 
import SUITPy.flatmap as flatmap 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.stats.multitest as smm
from sklearn.preprocessing import StandardScaler

# Prevent scientific notation
pd.set_option('display.float_format', '{:.3f}'.format)

# Set directory
data_dir = '/project/normative_cerebellum'

# Import data
# Note: We selected subjects who were included in the normative model and had complete behavioral scores
data = pd.read_csv(os.path.join(data_dir, 'behavioral_outcomes/zscores_behaviors.csv'))

behavs = data.iloc[:, 1:11] # Behavioral scores
parcels = data.iloc[:, 14:] # Parcel normative z-scores
age = data['age']
sex = data['sex']

##################################
### MASS-UNIVARIATE REGRESSION ###
##################################

# List to collect results
results = []

# Standardize age and sex (the parcels are already z-scored)
scaler = StandardScaler()
data[['age', 'sex']] = scaler.fit_transform(data[['age', 'sex']])  

# Loop through each DV (behavior)
for dv_name in behavs.columns:
    y = data[dv_name].copy()  
    
    # Loop through each parcel
    for parcel_name in parcels.columns:
        X = data[[parcel_name]].copy()  # Use only this parcel
        X = sm.add_constant(X)  # Add intercept
        X['age'] = age  # Add age
        X['sex'] = sex  # Add sex
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Append the results as a dict
        results.append({
            'DV': dv_name,
            'Parcel': parcel_name,
            'Coefficient': model.params[parcel_name],
            'P-value': model.pvalues[parcel_name]
        })

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Regression betas from long to wide
coef_df = results_df.pivot(index='DV', columns='Parcel', values='Coefficient')
coef_df.index.name = None
coef_df.columns.name = None

# P-values from long to wide
pval_df  = results_df.pivot(index='DV', columns='Parcel', values='P-value')
pval_df.index.name = None
pval_df.columns.name = None

# Create significance mask (p < .05)
pval_mask_df = pval_df < 0.05

# Mask significant regression betas (set non-significant to 0)
coef_df[~pval_mask_df] = 0

#######################
### PLOT ON FLATMAP ###
#######################

# Create cerebellum template based on fusion atlas
c_path = os.path.join(data_dir, 'atlases', 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_dseg.nii')
cerebellum = nib.load(c_path)
data_c = cerebellum.get_fdata()
region_labels = np.unique(data_c)

# Create a dataframe of region names
lab_num = pd.read_csv(os.path.join(data_dir, 'atlas_labels', 'fusion_labels.txt'), header=None)  
lab_num.columns = ['parcel_name']
lab_num['parcel'] = range(1, 33)

# Select socio-linguistic behaviors (since they were significant after PLS bootstrapping)
coef_df = coef_df.loc[['reading', 'lang_compr', 'srs']]

# Loop through each behavior and create a new flatmap
for index, row in coef_df.iterrows():

    # Create a copy of the cerebellum template to modify for each behavior
    new_img = np.zeros_like(data_c)
    
    # Map each value to the corresponding cerebellum region
    for i, region in enumerate(region_labels):
        if region != 0:  # Skip region 0 (background)
            new_img[data_c == region] = row[i-1]  

    # Check if we have any signigicant values
    if np.all(new_img == 0):
        print(f"No significant values to plot for {index}, skipping...")
        continue

    # Use behavior as part of the filename
    beh_name = index
    
    # Save image 
    img = nib.Nifti1Image(new_img, cerebellum.affine)
    nib.save(img, os.path.join(data_dir, 'behavioral_outcomes', f'{beh_name}_zscore_regression.nii.gz'))
    print(f'Saved image for {beh_name} task')

    # Plot on flatmap 
    c_map = flatmap.vol_to_surf(img) 
    c_fig = flatmap.plot(data=c_map, cmap='inferno', render='matplotlib', 
                         space='MNISymC', colorbar=True, new_figure=True)

    # Save figure
    plt.savefig(os.path.join(data_dir, 'behavioral_outcomes', f'{beh_name}_zscore_regression.png'), dpi=300)  