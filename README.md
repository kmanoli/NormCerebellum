## Cerebellar growth is associated with domain-specific cerebral maturation and behavioral outcomes 

### This repository contains analysis scripts and cerebellar normative models presented in this manuscript: TBD

### Instructions
If you'd like to reproduce the analyses in the manuscript, you can run the following scripts:

#### A. Cerebellar segmentation
Scripts in **scripts/1_segmentation** generate anatomical (ACAPULCO: https://www.sciencedirect.com/science/article/pii/S1053811920303062) and functional (https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master) cerebellar parcellations in individual subjects' space. 
- **anat/1_run_acapulco.sh**: Runs ACAPULCO (via Singularity).
- **func/1_fsl_func_warp.sh**: Warps MNI-aligned MDTB, functional fusion, and resting-state cerebellar atlases to subject space (via FSL).
- **<parcellation_type>/2_<parcellation_type>_native_vols.py**: Counts volumes per parcel. Functional parcellations include outlier detection (anatomical ACAPULCO masks are manually corrected when necessary via ITK-SNAP: https://www.itksnap.org/pmwiki/pmwiki.php). 

_Expected outputs_: Dataframe and images of subject-space anatomical and functional parcellations. 

_Expected runtime_: ~15 minutes per atlas per subject (manual corrections: ~3 hours per subject).

#### B. Normative modeling
Scripts in **scripts/2_normative_modeling** generate normative models based on the Hierarchical Bayesian Regression algorithm in the PCN toolkit (https://github.com/amarquand/PCNtoolkit). Sex and site were modeled as batch (random) effects.
- **1_qc_euler_dk.py**: Performs QC based on the Euler index of cerebral cortex surface reconstructions (for consistency, we included the same subjects in cerebral and cerebellar normative models).
- **2_data_prep.py**: Prepares and splits the data for normative modeling.
- **3_norm_modeling_hbr.py**: Fits parcel-wise linear and 3rd-order b-spline normative models, including model diagnostics and quantile plots. 
- **4_loocv.py**: Performs model comparison of linear and b-spline models via leave-one-out cross-validation, as well as additional odel diagnostics.

_Expected outputs_: Parcel-wise normative models, model diagnostics, and quantile plots.

_Expected runtime_: ~3 minutes per parcel per model type (linear or b-spline). This might vary based on computational infrastructure, model algorithm, and batch effects.

<img width="955" height="576" alt="Screenshot 2025-08-03 at 20 38 44" src="https://github.com/user-attachments/assets/ab0e8b5b-82bb-45bc-9934-403eecf13903" />

#### C. Associations with cerebral cortex
Scripts in **scripts/3_cerebral_associations** perform associations of growth trajectories between the cerebellum and the cerebral cortex (Desikan-Killiany atlas) via regularized regression.
- **cerebral_assoc.py**: Compares regularization methods, uses the winning method for cerebro-cerebellar associations, performs permutation significance testing, and plots FDR-corrected associations.

_Expected outputs_: Heatmap and cerebral cortex plots of significant cerebro-cerebellar associations.

_Expected runtime_: ~2 hours per cerebellar atlas.

<img width="665" height="642" alt="Screenshot 2025-08-03 at 20 35 51" src="https://github.com/user-attachments/assets/ca84e5a5-ec87-4c44-b04f-6ba6e403ae46" />

#### D. Associations with behavioral outcomes
Scripts in **scripts/4_behavioral_outcomes** perform associations of cerebellar growth trajectories and individual differences in behavioral markers.
- **1_behav_pls.py**: Performs multivariate association of cerebellar parcels and behavioral outcomes via Partial Least Squares (PLS) analysis. Selects number of components (latent variables), and performs cross-validation, permutation significance testing, and bootstrapping of the PLS model.
- **2_behav_regressions.py**: Unpacks the PLS results via univariate association of cerebellar parcels and behavioral outcomes (mass-univariate linear regression).
- **3_behav_cereb_cortex.py**: Compares the relative contribution of cerebellar and cerebral (or combined) growth to behavioral outcomes via regularized regression. Compares regularization methods across parcel set combinations, and then compares model performance of the winning method for each parcel set combination across behaviors. 

_Expected outputs_: Plots of cerebellar-behavioral associations.

_Expected runtime_: ~30 minutes per cerebellar atlas.

<img width="530" height="638" alt="Screenshot 2025-08-03 at 20 27 17" src="https://github.com/user-attachments/assets/83122918-997e-49ac-9548-d0081148ab3c" />

### Data
This study is based on openly accessible data from the Lifespan Baby Connectome Project (BCP; nda.nih.gov/edit_collection.html?id=2848) and the Lifespan 2.0 Human Connectome Project in Development (HCP-D; nda.nih.gov/general-query.html?q=query=featured-datasets:HCP%20Aging%20and%20Development). Cerebellar atlases are available here: https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master.

Linear and b-spline normative models for cerebellar anatomical and functional parcels are available in the **normative_models** folder. Labels for cerebral and cerebellar anatomical and functional parcellations are available in the **atlas_labels** folder.

### Requirements and installation
All Python scripts were executed using Python 3.11.8 (https://www.python.org/downloads/) on macOS (please see associated scripts for required Python libraries). The ACAPULCO container was downloaded from https://gitlab.com/shuohan/acapulco and executed via Singularity (https://docs.sylabs.io/guides/3.0/user-guide/installation.html). Functional cerebellar segmentations were performed via FSL (https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index). Normative modeling was implemented via the PCN toolkit version 0.29 (https://github.com/amarquand/PCNtoolkit). To clone this repositiry, run: git clone https://github.com/kmanoli/NormCerebellum.git

The scripts can be run on a standard desktop computer, however, we recommend running these analyses (especially segmentation algorithms and normative models) on a cluster to parallelize computations.



