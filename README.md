# Code for Master Thesis 
## Master’s Program: Data Analysis & Machine–Statistical Learning 
### University of Crete (UOC) & Foundation for Research & Technology Hellas (FORTH)

## Overview

This repository contains the code developed for my Master’s thesis:

**“Uncertainty Quantification in Deep Learning: Methods and Applications in MRI Prostate Data.”**

The project presents an experimental analysis of multiple uncertainty quantification and machine learning methods applied to **prostate MRI data**. The evaluated approaches include:

- Single-layer Exact Gaussian Process Regression  
- Deep Kernel Learning  
- Gaussian Process Classification  
- Logistic Regression  
- K-Nearest Neighbors  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

The MRI prostate dataset is derived from the study:

> *Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study*, Anindo Saha et al.
> https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext

The main objective of this work is to **identify models that achieve low predictive uncertainty** while accurately predicting **ISUP grade groups**.

The repository includes:
- Reusable Python modules for model implementation  
- Jupyter notebooks for experiments, demonstrations and analysis  
- Data preprocessing and exploratory data analysis   

---


## Dependencies

This project relies on the following scientific Python and machine‑learning libraries:

- NumPy  
- Pandas  
- SciPy  
- Scikit‑learn  
- imbalanced‑learn  
- Optuna  
- XGBoost  
- LightGBM  
- CatBoost  
- Matplotlib  
- Seaborn  

Additional project‑specific modules:

- PyTorch  
- GPytorch  
- `deep_gp` (custom package for data loading and evaluation)

 

Parts of the code are adapted and modified from the official **GPyTorch documentation**:  
https://docs.gpytorch.ai/en/stable/

---

### Data Preprocessing

In all notebooks except **05-new_approach.ipynb**, radiomic features and clinical labels are merged using `study_id`, producing a unified dataset containing predictors and ISUP grades.

The following preprocessing pipeline is used in the Exact Gaussian Process Regression, Deep Gaussian Processes, and the Binary Classification scheme (ISUP grades 0–5):

#### **Class‑0 Undersampling via Nearest Neighbors**
To reduce the dominance of ISUP class 0, each minority‑class sample (ISUP 1–5) is paired with its nearest class‑0 neighbor.  
Only these selected class‑0 samples are retained, producing a more balanced dataset before oversampling.

#### **SMOTE Oversampling**
SMOTE is applied **only** to ISUP classes 3, 4, and 5, increasing each to 150 samples while leaving classes 0–2 unchanged.

After undersampling and oversampling, a train–test split is applied to this final dataset before model training.

---

### **New Classification Approach (05-new_approach.ipynb)**

A new preprocessing strategy is introduced in the notebook **05-new_approach.ipynb**:

- No undersampling of ISUP grade 0 is performed.  
- The dataset is used in its original imbalanced form.
- SMOTE is applied only inside the nested cross‑validation and only on the training folds , ensuring no information leakage.

Additionally, a univariate feature‑selection step is performed on the full dataset using the Mann–Whitney U test with Benjamini–Hochberg correction to control the false discovery rate (FDR).  
Only statistically significant features are retained for modelling.


---
### Notebooks: 

1. **GP_demo folder**
   - Implements Gaussian Processes (GPs) from scratch using NumPy and Matplotlib.
   - Visualizes samples from the GP prior and posterior.

2. **statistics**
   - Provides a description of the MRI prostate dataset.
   - Includes exploratory data analysis and statistical summaries.

3. **activation_functions**
   - Visualizes several activation functions used in neural networks.
   - Corresponds to the Neural Networks chapter of the thesis.

4. **GPtorch_example folder**
    - A tutorial for GP regression form the GPyTorch documentation.
    
5. **Classification_methods**
    - Binary and multi-class classification experiments.
    - Studies the effect of removing class 0 and handling class imbalance.
    - Studies the effect of applying SMOTE inside cross-validation and only in the training folds
    - Implements and compares multiple classical ML classifiers.
    

5. The notebooks numbered from **00 to 04** contain the **core implementation and experimental code** used in this project. These notebooks include the main modeling pipelines, training procedures and evaluation workflows:
    - **00** – Linear regression  
    - **01** – Deep Kernel Learning  
    - **02** – Deep Kernel Learning (extended experiments)
    - **03** – Exact single-layer Gaussian Processes  
    - **04** – Deep Gaussian Processes  
