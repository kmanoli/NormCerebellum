#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: ACAPULCO automatically extracts lobular volumes in MNI space. Here, we are using each subject's native space ACAPULCO mask to manually calculate lobular volumes. These masks were visually inspected and, if necessary, manually corrected with ITK-SNAP.

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

# Set project directory
project_dir = '/data/normative_cerebellum'

# Import subject list
with open(os.path.join(project_dir, 'all_subjects.txt'), 'r') as file:
    subject_list = [line.strip() for line in file]

# Get volume label names from example ACAPULCO MNI segmentation (standard ACAPULCO output)
vol_names = pd.read_csv(os.path.join(
    project_dir, 'segmentations/acapulco/output/HCD0001305_V1_MR/T1w_acpc_dc_restore_n4_mni_seg_post_volumes.csv'
))
vol_names = vol_names.iloc[1:, 0:1].reset_index(drop=True)
print(vol_names)

# Initialize an empty DataFrame to store native space volumes
vols_df = pd.DataFrame(columns=['subject', 'label', 'volume'])

# Loop through subjects
for subject in subject_list:
    mask_dir = os.path.join(project_dir, 'segmentations/acapulco/output', subject)
    mask_file = glob.glob(os.path.join(mask_dir, 'T1w_acpc_dc_restore_n4_mni_seg_post_inverse.nii*')) # Native space ACAPULCO mask
    
    if mask_file:
        mask_img = nib.load(mask_file[0])
    
    # Calculate volumes per lobule
    mask_data = np.round(mask_img.get_fdata(dtype=np.float32)).astype(int)
    voxel_volume = np.prod(mask_img.header.get_zooms())
    unique_labels = np.unique(mask_data)
    
    parcel_volumes = {
        label: round(np.sum(mask_data == label) * voxel_volume, 2)
        for label in unique_labels if label != 0 # 0 represents background
    }
    
    vols = pd.DataFrame.from_dict(parcel_volumes, orient='index', columns=['volume'])
    vols.reset_index(inplace=True)
    vols.rename(columns={'index': 'label'}, inplace=True)
    vols['subject'] = subject
    vols = pd.concat([vols, vol_names], axis=1)
    vols_df = pd.concat([vols_df, vols], ignore_index=True)

# Check for missing values
missing_values = vols_df[vols_df['volume'].isnull()]['subject']
print("Rows with missing data:")
print(missing_values)

# Convert to wide format
wide_df = vols_df.pivot_table(index='subject', columns='name', values='volume', aggfunc='first')
wide_df.reset_index(inplace=True)

# Save output
wide_df.to_csv(os.path.join(project_dir, 'normative_modeling/anat_native/input/all_anat_native_vols.csv'), index=False)

