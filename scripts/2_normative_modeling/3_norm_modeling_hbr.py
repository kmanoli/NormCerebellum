#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This file uses anatomical lobules as an example input, but the process is the same for functional parcel volumes.

import os
import pandas as pd 
import pcntoolkit as ptk 
import numpy as np 
import pickle
from matplotlib import pyplot as plt 
import seaborn as sns 
sns.set_style('darkgrid')

###########################
### MODEL SPECIFICATION ###
###########################

# Set directories
base_dir = "/project/normative_cerebellum/normative_modeling/anat_native"
main_dir = os.path.join(base_dir, "output")

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

# Set up and run model for every parcel
for parc in parcels:
    # Loop through both model types
    for model_type in ['linear', 'bspline']:
        print(f"Processing {parc} with {model_type} model")
        
        # Set up processing directory 
        processing_dir = os.path.join(main_dir, parc, f'{model_type.capitalize()} Model')
        if not os.path.isdir(processing_dir):
            os.makedirs(processing_dir)
            
        # Set up figures directory
        figures_dir = os.path.join(processing_dir, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

	# Prepare input data 
        X_train = (pd.to_numeric(train["age"])).to_numpy(dtype = float)
        X_test = (pd.to_numeric(test["age"])).to_numpy(dtype = float)
        Y_train = train[parc].to_numpy(dtype = float)
        Y_test = test[parc].to_numpy(dtype = float)
        batch_effects_train = train[['sex', 'site']].to_numpy(dtype = int)
        batch_effects_test = test[['sex', 'site']].to_numpy(dtype = int)

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
        log_dir = os.path.join(processing_dir, 'log/')

        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        
        # Model specification 
        likelihood='SHASHb'
        outputsuffix = '_estimate'      
        inscaler_type='standardize'
        outscaler_type='standardize'

        inscaler = ptk.util.utils.scaler(inscaler_type)
        X_train_standardized = inscaler.fit_transform(X_train)
        X_test_standardized = inscaler.transform(X_test)

        outscaler = ptk.util.utils.scaler(outscaler_type)
        Y_train_standardized = outscaler.fit_transform(Y_train)
        Y_test_standardized = outscaler.transform(Y_test)

        n_mcmc_samples = 2000 
        n_tuning_samples = 500 
        n_chains = 4
        n_cores = 8
        target_accept = 0.99
	
	# Run the model
        ptk.normative.fit(covfile=covfile,
                           respfile=respfile,
                           trbefile=trbefile,
                           testcov=testcovfile_path,
                           testresp=testrespfile_path,
                           tsbefile=tsbefile,
                           log_path=log_dir,
                           saveoutput='True',
                           output_path=output_path, 
                           savemodel='True',
                           binary='True',
                           outputsuffix=outputsuffix,
                           alg='hbr',
                           n_samples=n_mcmc_samples,
                           n_tuning=n_tuning_samples,
                           n_chains=n_chains,
                           cores=n_cores,
                           target_accept=target_accept,
                           inscaler=inscaler_type,
                           outscaler=outscaler_type,
                           likelihood=likelihood,
                           model_type=model_type,
                           linear_mu='True',
                           random_mu = 'True', 
                           random_intercept_mu='True', 
                           random_slope_mu = 'True', 
                           random_sigma = 'True', 
                           centered_sigma = 'True', 
                           random_intercept_sigma ='True', 
                           random_slope_sigma = 'True'
                     )

        # Load the model and show the idata
        model_path = os.path.join(output_path, f'NM_0_0{outputsuffix}.pkl')
        nm = pickle.load(open(model_path,'rb'))
        nm.hbr.idata.posterior
        
        ###################
        ### DIAGNOSTICS ###
        ###################
        
        # Rhat
        Rhat = nm.hbr.Rhats()
        for r in Rhat.keys():
            plt.plot(Rhat[r], color = 'slateblue')
            plt.title(r)
            plot = os.path.join(figures_dir , f'{parc}_{r}_{model_type}.pdf')
            plt.savefig(plot)
            plt.close()

        # We need to provide a matrix of shape [N,d]. In this case d=1
        X_test_standardized_np = np.array(X_test_standardized)[:,np.newaxis]
        Y_test_standardized_np = np.array(Y_test_standardized)[:,np.newaxis]
        batch_effects_test2 = batch_effects_test.astype(int)
        
        with open('tsbefile2.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(batch_effects_test2), file)
        tsbefile2 = os.path.join(processing_dir, 'tsbefile2.pkl') 
        
        X_train_standardized_np = np.array(X_train_standardized)[:,np.newaxis]
        Y_train_standardized_np = np.array(Y_train_standardized)[:,np.newaxis]
        batch_effects_train2 = batch_effects_train.astype(int)
        
        with open('trbefile2.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(batch_effects_train2), file)
        trbefile2 = os.path.join(processing_dir, 'trbefile2.pkl') 
        
        # Extract z-scores
        zs = nm.get_mcmc_zscores(X_test_standardized_np, Y_test_standardized_np, tsbefile=tsbefile2)
        zs_df = pd.DataFrame(zs)
        zs_df.columns = ['z_score']
        zs_df.to_csv(os.path.join(main_dir, f'{parc}/zscores_{model_type}_{parc}.csv'))      
        
        # Diagnostics based on z-scores
        bes = np.unique(batch_effects_test, axis = 0)
        for ibe, be in enumerate(bes):
            idx = (batch_effects_test == be).all(1)
            this_zs = zs[idx]
            zs_sorted = np.sort(this_zs)
            ns = np.random.randn(this_zs.shape[0])
            plt.scatter(np.sort(ns), zs_sorted+2*ibe)
            plt.title('Q-Q Plot of z-scores')
            
        plot = os.path.join(figures_dir , f'{parc}_Q-Q_plot_hbr_{model_type}.pdf')
        plt.savefig(plot)
        plt.close()
        
        #########################
        ### PLOT TRAJECTORIES ###
        #########################
        
        # Get the MCMC-estimated quantiles
        minx = np.min(X_test_standardized)
        maxx = np.max(X_test_standardized)

        #zscores = np.arange(-3,4)[:,np.newaxis]
        zscores = np.arange(-3,4)
        
        n_synthetic_samples = 200
        synthetic_X = np.linspace(minx, maxx, n_synthetic_samples)[:,np.newaxis]

        be = np.zeros((n_synthetic_samples,1))
        q = nm.get_mcmc_quantiles(synthetic_X,be,zscores)

        # Plot the quantiles
        sns.set_style('ticks')

        plt.scatter(X_test, Y_test, s = 5, alpha = 0.6, c = 'peachpuff', edgecolors = None)
        sns.despine(right = True, top = True)
        for i, v in enumerate(zscores):
        
            if v in [-1, 1]:
                plt.fill_between(inscaler.inverse_transform(synthetic_X).flatten(),
                                 outscaler.inverse_transform(q[i]),
                                 outscaler.inverse_transform(q[i+1]),
                                 color = 'darkorange',
                                 edgecolor = 'none',
                                 alpha = 0.1)
                continue
        
            thickness = 1
            linestyle = "-"
        
            if v == 0:
                thickness = 1.6
            if abs(v) > 2:
                linestyle = "--"
            
            plt.title(f'{parc}')
            plt.xlabel('Age')
            plt.ylabel('Volume (mm3)')
            plt.plot(inscaler.inverse_transform(synthetic_X), 
                     outscaler.inverse_transform(q[i]), 
                     linewidth = thickness, 
                     linestyle = linestyle, 
                     color = 'black', 
                     alpha = 0.7)
        
            if i != len(zscores) -1:
                plt.fill_between(inscaler.inverse_transform(synthetic_X).flatten(),
                                 outscaler.inverse_transform(q[i]),
                                 outscaler.inverse_transform(q[i+1]),
                                 color = 'darkorange',
                                 edgecolor = 'none',
                                 alpha = 0.1)
        
        plot = os.path.join(figures_dir , f'{parc}_quantile_plot_hbr_{model_type}.pdf')
        plt.savefig(plot)
        plt.close()
