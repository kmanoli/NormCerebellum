#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""
# NOTE: This script identifies subjects with problematic FreeSurfer cerebral cortex surface reconstruction (surface holes or Euler index). These subjects are removed from cerebral and cerebellar normative models to keep subject numbers consistent for cerebro-cerebellar comparisons.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# QC functions
def get_surface_holes(subfile, directory, file, line_index):
    all_surfholes = []
    
    subs = pd.read_csv(subfile, header=None)
    subslist = list(subs[0])
    
    for sub in subslist:
        os.chdir(os.path.join(directory, sub))
        with open(file, 'rt') as in_fh:
            lines = in_fh.readlines()
        
        holes = []
        
        for item in lines[line_index]:
            for subitem in item.split():
                if (subitem.isdigit()):
                    holes.append(subitem)
        
        holes = ''.join(holes)
        holes = int(holes)
        all_surfholes.append(holes)
    
    return all_surfholes

def plot_distribution(all_surfholes):
    plt.hist(all_surfholes, bins=len(all_surfholes), label='Data')
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    plt.show()

def find_high_surfholes(subslist, all_surfholes, threshold):
    df_unflitered = pd.DataFrame(list(zip(subslist, all_surfholes)), columns=['subject_ID', 'surfholes'])
    df_holes = df_unflitered[df_unflitered['surfholes'] > threshold]
    df_holes = df_holes.drop('surfholes', axis=1)
    num_subs = df_holes.shape[0]
    return df_unflitered, df_holes, num_subs
    
# Set directory 
project_dir = '/data/normative_cerebellum'

# Configuration for HCPD data
hcpd_directory = os.path.join(project_dir, 'segmentations/dk/hcpd_euler')
hcpd_subfile = os.path.join(project_dir, 'hcpd_subjects.txt')
hcpd_file = 'aseg.stats'
hcpd_line_index = 33  # The line number where surface holes data is found in above file
hcpd_threshold = 80 # Based on distribution tail (we defined this post-hoc after plotting every dataset distribution)

# Configuration for BCP data
bcp_directory = os.path.join(project_dir, 'segmentations/dk/bcp_euler')
bcp_subfile = os.path.join(project_dir, 'bcp_subjects.txt')
bcp_file = 'aseg.stats'
bcp_line_index = 32  # The line number where surface holes data is found in above file
bcp_threshold = 900 # Based on distribution tail (we defined this post-hoc after plotting every dataset distribution)

# Process HCPD data
hcpd_subs = pd.read_csv(hcpd_subfile, header=None)
hcpd_subslist = list(hcpd_subs[0])

hcpd_surfholes = get_surface_holes(hcpd_subfile, hcpd_directory, hcpd_file, hcpd_line_index)
plot_distribution(hcpd_surfholes)
hcpd_unfiltered, hcpd_holes, hcpd_num_subs = find_high_surfholes(hcpd_subslist, hcpd_surfholes, hcpd_threshold)
hcpd_holes.to_csv(os.path.join(project_dir,'normative_modeling/dk_native/input/bad_euler_subs_hcpd.csv', index=False)
print(f"The number of HCPD subjects with more than {hcpd_threshold} surfholes is: {hcpd_num_subs}")

# Process BCP data
bcp_subs = pd.read_csv(bcp_subfile, header=None)
bcp_subslist = list(bcp_subs[0])

bcp_surfholes = get_surface_holes(bcp_subfile, bcp_directory, bcp_file, bcp_line_index)
plot_distribution(bcp_surfholes)
bcp_unfiltered, bcp_holes, bcp_num_subs = find_high_surfholes(bcp_subslist, bcp_surfholes, bcp_threshold)
bcp_holes.to_csv(os.path.join(project_dir, 'normative_modeling/dk_native/input/bad_euler_subs_bcp.csv', index=False)
print(f"The number of BCP subjects with more than {bcp_threshold} surfholes is: {bcp_num_subs}")

# Rescale surfholes and clean datasets combined

# Combine surfholes
holes_hcpd = hcpd_unfiltered.iloc[:, 1]
holes_bcp = bcp_unfiltered.iloc[:, 1]
combined_df = pd.concat([holes_hcpd, holes_bcp], ignore_index=True)
combined_df = combined_df.to_frame()

# Rescale and plot distribution
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_df)
plot_distribution(scaled_data)

# Find high surfholes
subs_hcpd = hcpd_unfiltered.iloc[:, 0]
subs_bcp = bcp_unfiltered.iloc[:, 0]
combined_subs = pd.concat([subs_hcpd, subs_bcp], ignore_index=True)
all_threshold = 0.5 # Based on distribution tail

all_unfiltered, all_holes, all_num_subs = find_high_surfholes(combined_subs, scaled_data, all_threshold)
all_holes.to_csv(os.path.join(project_dir, 'normative_modeling/dk_native/input/bad_euler_subs.csv', index=False)
print(f"The number of subjects with more than {all_threshold} surfholes is: {all_num_subs}")
