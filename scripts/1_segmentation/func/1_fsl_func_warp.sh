#!/bin/bash

# Set base directories
project_dir="/project/normative_cerebellum"
input_dir="${project_dir}/data"
output_dir="${project_dir}/segmentations"

# Subject list
subject_list="${project_dir}/hcpd_subjects.txt"

# Atlas names 
atlas_names=(
  "fusion"
  "MDTB"
  "rest"
)

# Atlas files (in same order as atlas names)
atlases=(
  "atl-NettekovenSym32_space-MNI152NLin2009cSymC_dseg.nii"
  "atl-MDTB10_space-MNI_dseg.nii"
  "atl-Buckner7_space-MNI_dseg.nii"
)

# Loop over each subject
while IFS= read -r subject; do
  echo "Processing subject: ${subject}"
  mkdir -p "${output_dir}/${atlas_name}/${subject}"

  # Loop over atlases and their corresponding output names
  for i in "${!atlases[@]}"; do
    atlas_file="${atlases[$i]}"
    atlas_name="${atlas_names[$i]}"
    
    # Warp MNI-aligned atlas to native space 
    applywarp \
      --ref="${input_dir}/${subject}/T1w_acpc_dc_restore.nii.gz" \
      --in="${project_dir}/atlases/${atlas_file}" \
      --out="${output_dir}/${atlas_name}/${subject}/${atlas_name}_native.nii.gz" \
      --warp="${project_dir}/HCPD_xfms/${subject}/standard2acpc_dc.nii.gz" \ # Deformation file
      --interp=nn
  done

done < "$subject_list"

