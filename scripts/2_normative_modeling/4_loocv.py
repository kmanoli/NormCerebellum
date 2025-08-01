#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This file uses anatomical lobules as an example input, but the process is the same for functional parcel volumes.

import os
import csv
import time
import pandas as pd  
from IPython.display import display
import arviz as az
import pcntoolkit as ptk  
import numpy as np  
import pickle
from matplotlib import pyplot as plt  
import seaborn as sns  
import scipy.stats as sst
import pymc as pm
from itertools import product
from pcntoolkit.normative_model.norm_hbr import NormHBR
sns.set_style("whitegrid")

# Set directories
base_dir = "/project/normative_cerebellum/normative_modeling/anat_native"
main_dir = os.path.join(base_dir, "loocv")

# Load train and test data
test = pd.read_csv(os.path.join(base_dir, "input/test_anat_native_vols_clean.csv"))
train = pd.read_csv(os.path.join(base_dir, "input/train_anat_native_vols_clean.csv"))

col_name = list(test.columns.values)

# Remove extra columns to retain only parcel names
rmv = ['subject', 'sex', 'age', 'site']
for val in rmv:
    if val in col_name:
        col_name.remove(val)
parcels = col_name

# Initialize result storage
linear_diag = []
bspline_diag = []
comp_df = []

# Set up and run LOOCV for every parcel
for parc in parcels:
    # Store LOOCV results for comparison
    loo_results = {}
    
    # Loop through both model types
    for model_type in ['linear', 'bspline']:
        
        # Set up processing directory for loading models
        processing_dir = os.path.join(base_dir, parc, f'{model_type.capitalize()} Model')
        if not os.path.isdir(processing_dir):
            os.makedirs(processing_dir)
        
        # Set up LOOCV figures directory
        figures_dir = os.path.join(main_dir, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        # Configure input data 
        X_train = (pd.to_numeric(train["age"])).to_numpy(dtype=float)
        X_test = (pd.to_numeric(test["age"])).to_numpy(dtype=float)
        Y_train = train[parc].to_numpy(dtype=float)
        Y_test = test[parc].to_numpy(dtype=float)
        batch_effects_train = train[['sex', 'site']].to_numpy(dtype=int)
        batch_effects_test = test[['sex', 'site']].to_numpy(dtype=int)

        for name, data in [('X_train', X_train), ('Y_train', Y_train), 
                          ('X_test', X_test), ('Y_test', Y_test),
                          ('trbefile', batch_effects_train), 
                          ('tsbefile', batch_effects_test)]:
            with open(f'{name}.pkl', 'wb') as file:
                pickle.dump(pd.DataFrame(data), file)

        respfile = os.path.join(processing_dir, 'Y_train.pkl')
        covfile = os.path.join(processing_dir, 'X_train.pkl')
        testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')
        testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')
        trbefile = os.path.join(processing_dir, 'trbefile.pkl')
        tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')
        output_path = os.path.join(processing_dir, 'Models/')
        
        outputsuffix = '_estimate'      
        inscaler_type='standardize'
        outscaler_type='standardize'

        inscaler = ptk.util.utils.scaler(inscaler_type)
        X_train_standardized = inscaler.fit_transform(X_train)
        X_test_standardized = inscaler.transform(X_test)

        outscaler = ptk.util.utils.scaler(outscaler_type)
        Y_train_standardized = outscaler.fit_transform(Y_train)
        Y_test_standardized = outscaler.transform(Y_test)
        
        # Load the model
        model_path = os.path.join(output_path, f'NM_0_0{outputsuffix}.pkl')
        nm = pickle.load(open(model_path,'rb'))
                   
        # Compute LOO
        with nm.hbr.get_model(X_train_standardized, Y_train_standardized, batch_effects_train):
            pm.compute_log_likelihood(idata=nm.hbr.idata)
            current_loo = az.loo(nm.hbr.idata)
            print(f"{model_type.capitalize()} LOO for {parc}:")
            print(current_loo)
            loo_results[model_type] = current_loo
            
        # Store diagnostics
        if model_type == 'linear':
            linear_diag.append(current_loo)
        else:
            bspline_diag.append(current_loo)
            
        # Plot trace
        az.plot_trace(nm.hbr.idata, var_names="~samples", filter_vars="like")
        plot = os.path.join(figures_dir, f'{parc}_trace_{model_type}.png')
        plt.savefig(plot)
        plt.show()

        # Plot forest
        az.plot_forest(nm.hbr.idata, var_names="~samples", filter_vars="like")
        plot = os.path.join(figures_dir, f'{parc}_forest_{model_type}.png')
        plt.savefig(plot)
        plt.show()

        # Plot autocorr
        az.plot_autocorr(nm.hbr.idata, var_names="~samples", filter_vars="like")
        plot = os.path.join(figures_dir, f'{parc}_autocorr_{model_type}.png')
        plt.savefig(plot)
        plt.show()

        # Prior and posterior checks
        backup_idata = nm.hbr.idata
        nm.hbr.sample_prior_predictive(
            X_test_standardized,
            batch_effects_test,
            samples=1000,
            idata=nm.hbr.idata,
        )
        az.plot_ppc(nm.hbr.idata, num_pp_samples=100, group="prior")
        plot = os.path.join(figures_dir, f'{parc}_prior_{model_type}.png')
        plt.savefig(plot)
        plt.show()
        
        nm.hbr.idata = backup_idata
        nm.predict(X_test_standardized, tsbefile=tsbefile)
        az.plot_ppc(nm.hbr.idata, num_pp_samples=100)
        plot = os.path.join(figures_dir, f'{parc}_posterior_{model_type}.png')
        plt.savefig(plot)
        plt.show()

    # Model comparison after both models are processed
    df_comp_loo = az.compare(loo_results, ic="loo")
    print(f"Model comparison for {parc}:")
    print(df_comp_loo)
    comp_df.append(df_comp_loo)
    
    # Create comparison plot 
    comp_figures_dir = os.path.join(figures_dir, "Model Comparisons") 
    if not os.path.exists(comp_figures_dir):
        os.makedirs(comp_figures_dir)
        
    az.plot_compare(df_comp_loo, insample_dev=False)
    plot = os.path.join(comp_figures_dir, f'{parc}_model_comp.png')
    plt.savefig(plot)
    plt.show()

print("Final Results:")
print("Comparison DataFrames:", comp_df)
print("Linear Diagnostics:", linear_diag)
print("Bspline Diagnostics:", bspline_diag)  

# Save results
comp_df_ = pd.concat(comp_df, ignore_index=False)
comp_df_.to_csv(os.path.join(main_dir, 'model_comparison.csv'))

linear_diag_ = pd.concat(linear_diag, ignore_index=False)
linear_diag_.to_csv(os.path.join(main_dir, 'linear_diagnostics.csv'))

bspline_diag_ = pd.concat(bspline_diag, ignore_index=False)
bspline_diag_.to_csv(os.path.join(main_dir, 'bspline_diagnostics.csv'))
