#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This file uses anatomical lobules as an example input, but the process is the same for functional parcel volumes.

import pandas as pd
from sklearn.model_selection import train_test_split

# Set directory 
project_dir = '/data/normative_cerebellum'

# Load data
dat = pd.read_csv(os.path.join(project_dir, 'normative_modeling/anat_native/input/all_anat_native_vols.csv'))

# Remove bad DK subjects based on Euler index
bad_euler = pd.read_csv(os.path.join(project_dir, 'normative_modeling/dk_native/input/bad_euler_subs.csv')

to_remove = bad_euler['subject_ID'].tolist()
dat_clean = dat[~dat['subject'].isin(to_remove)]

# Add age, sex, and site variables
dem = pd.read_csv(os.path.join(project_dir, 'all_demographics.csv')

# Make sub ID columns consistent
dem = dem.rename(columns={'src_subject_id': 'subject'})

# Merge demographic information
all_dat = pd.merge(dat_clean, dem, on='subject', how='inner')

# Turn age into years
all_dat['interview_age'] = pd.to_numeric(all_dat['interview_age']) / 12
all_dat['interview_age'] = all_dat['interview_age'].round(2)
all_dat = all_dat.rename(columns={'interview_age': 'age'})

# Save to CSV
all_dat.to_csv('/data/p_02671/normative_katerina/anat_native/vols/input/all_anat_native_vols_clean.csv', index=False)

# Split data in training and test sets (80/20 split, proportional sex and site)
train_all, test_all = train_test_split(all_dat, test_size=0.2, stratify=all_dat[['sex', 'site']])

# Save to CSV
train_all.to_csv(os.path.join(project_dir, 'normative_modeling/anat_native/input/train_anat_native_vols_clean.csv'), index=False)
test_all.to_csv(os.path.join(project_dir, 'normative_modeling/anat_native/input/test_anat_native_vols_clean.csv'), index=False)
