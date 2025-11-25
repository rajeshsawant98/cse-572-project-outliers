# Insurance Premium Modeling via Manual and Unsupervised Risk Segmentation
CSE 572: Data Mining — Final Project

---

## Overview

This project implements a full end-to-end insurance premium modeling pipeline and evaluates how 
different risk segmentation strategies affect premium stability, fairness, and pricing performance.

We compare:

### **1. Manual Risk Index v2 (Actuarial / Rule-Based)**
A domain-informed scoring system using 8 actuarial dimensions  
(Age, Income, Smoking, Health, Credit, Claims, Vehicle Age, Exercise/Dependents).

### **2. K-Means Segmentation (Unsupervised)**
Clustering on standardized numerical features to discover latent customer groups.

### **3. LASSO Continuous Risk Index (Sparse Supervised Model)**
A linear model on log-premium to extract statistically meaningful risk weights.

Each segmentation method is then paired with three premium calculation models:

- **Rule-Based premium** (multiplicative actuarial factors)  
- **Ideal premium** (oracle noise-free banding)  
- **Random Forest premium** (nonlinear ML model)

The goal is to determine which segmentation approach yields  
**stable, interpretable, and actuarially aligned premiums**.

---

## Project Structure

```
insurance-project/
│
├── data/
│   └── processed CSVs used in notebooks
│
├── notebooks/
│   ├── 01_manual_risk_premium.ipynb
│   ├── 02_kmeans_risk_premium.ipynb
│   ├── 03_policy_evaluation.ipynb
│   ├── create_risk_index_v2.ipynb
│   ├── Data_Exploration.ipynb
│   ├── lasso_feature_selection.ipynb
│   └── risk_index_lasso.ipynb
│
├── visualizations/
│   ├── Delta Distribution Visualization.png
│   ├── manual_vs_kmeans_comparison.png
│   ├── premium_pricing_comparison.png
│   └── risk_class_analysis.png
│
├── comparison_summary.txt
├── premium_comparison_summary.txt
├── .gitignore
└── README.md
```

---

## Key Notebooks

### **1. 01_manual_risk_premium.ipynb**
Implements Manual Risk Index v2 and generates rule-based, ideal, and ML premiums.

### **2. 02_kmeans_risk_premium.ipynb**
Runs K-Means segmentation and computes associated premium models.

### **3. 03_policy_evaluation.ipynb**
Performs delta analysis, migration matrices, and generates comparison plots.

---

## Visualizations Included

- `manual_vs_kmeans_comparison.png`  
- `premium_pricing_comparison.png`  
- `risk_class_analysis.png`  
- `Delta Distribution Visualization.png`

---

## Installation & Setup

```
git clone https://github.com/rajeshsawant98/cse-572-project-outliers.git
cd insurance-project
pip install -r requirements.txt
```

---

## Summary of Findings

- Manual segmentation yields clear actuarial structure.  
- K-Means misclassifies ~75% of customers.  
- Significant premium leakage occurs under K-Means.  
- Machine learning pricing heavily depends on segmentation quality.

---

## Repository Link
Replace USERNAME with your GitHub ID:
https://github.com/rajeshsawant98/cse-572-project-outliers/

---

## Authors

Mrudul Patil
Rajesh Sawant  
Ashutosh Kumbhar  
Aditya Patil
Ashish Damale

Arizona State University — CSE 572 Data Mining
