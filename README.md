# Coding for master thesis

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

The MRI prostate dataset is derived from the study:

> *Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study* 
> Anindo Saha et al.
> https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext

The main objective of this work is to **identify models that achieve low predictive uncertainty** while accurately predicting **ISUP grade groups**.

The repository includes:
- Reusable Python modules for model implementation  
- Jupyter notebooks for experiments, demonstrations and analysis  
- Data preprocessing and exploratory data analysis   

---

## Dependencies

The project relies on common scientific Python and deep learning libraries, including:

- NumPy  
- SciPy  
- Matplotlib  
- PyTorch  
- GPytorch  
- Scikit-learn  

Parts of the code are adapted and modified from the official **GPyTorch documentation**:  
https://docs.gpytorch.ai/en/stable/

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

5. The notebooks numbered from **00 to 04** contain the **core implementation and experimental code** used in this project. These notebooks include the main modeling pipelines, training procedures and evaluation workflows.
    - **00** – Linear regression  
    - **01** – Deep Kernel Learning  
    - **02** – Classification models  
    - **03** – Exact single-layer Gaussian Processes  
    - **04** – Deep Gaussian Processes  
