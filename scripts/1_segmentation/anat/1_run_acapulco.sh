#!/bin/bash

# Set project directories 
project_dir="/project/normative_cerebellum"
input_dir="${project_dir}/data"
output_dir="${project_dir}/segmentations/acapulco"

# Subject list 
subject_list="${project_dir}/all_subjects.txt"

# Loop through each subject in the list and run ACAPULCO via Singularity container
while IFS= read -r subject; do
    echo "Processing subject: ${subject}"

    singularity run --cleanenv -B "${project_dir}":"${project_dir}" acapulco-0.2.1.sif \
        -i "${input_dir}/${subject}/T1w_acpc_dc_restore.nii.gz" \
        -o "${output_dir}/output/${subject}"

done < "$subject_list"

