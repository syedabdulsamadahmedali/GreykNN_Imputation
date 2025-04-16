# 🔍 Grey k-Nearest Neighbors (Grey k-NN) for Missing Data Imputation

This repository implements **Grey k-Nearest Neighbors (Grey k-NN)** — an advanced imputation method that uses **Grey Relational Analysis (GRA)** instead of traditional distance metrics to handle **numerical**, **categorical**, and **mixed** data with missing values.

The method is evaluated across **24 real-world datasets** using performance metrics like **Normalized Root Mean Square Error (NRMS)** and **Accuracy of Estimation (AE)**, and optimized for both small and large datasets using **GPU acceleration (CuPy)** and **parallel processing (Joblib)** respectively.

---

## 📌 Key Highlights

- ✅ Handles numerical, categorical, and mixed data types.
- 🚀 GPU acceleration with CuPy for small datasets.
- 🧩 Parallel CPU processing for large datasets (up to 494K rows).
- 📊 Evaluated on 24 benchmark datasets with synthetic missingness (1%–20%).
- 🧠 Uses Grey Relational Grades (GRG) instead of Euclidean distance.
- 💡 Performance metrics include NRMS and AE, logged to Excel for all runs.

---

## ⚙️ Core Concepts

### What is Grey Relational Analysis (GRA)?

Unlike k-NN that relies on Euclidean distance, **GRA** measures similarity using **Grey Relational Grades**, making it better suited for datasets with:
- Nonlinear patterns
- Mixed data types
- Missing values

### Grey k-NN Steps:
1. Partition each row into observed and missing parts.
2. Compute Grey Relational Grades (GRG) with respect to all complete rows.
3. Select top `k` neighbors with highest GRG.
4. Impute using:
   - Weighted average (for numerical)
   - Weighted mode (for categorical)

---

## 🛠️ Implementation Overview

| Feature                     | Small Datasets                  | Large Datasets                    |
|----------------------------|----------------------------------|-----------------------------------|
| Processing method          | GPU-accelerated (CuPy)          | Parallelized CPU (Joblib)         |
| Batch size                 | 200                             | 50                                |
| GRG similarity metric      | Yes                             | Yes                               |
| Data types supported       | Numeric, Categorical, Mixed     | Numeric, Categorical, Mixed       |
| Evaluation metrics         | NRMS, AE                        | NRMS, AE                          |
| Logging & output           | Imputed files + Excel results   | Imputed files + Excel results     |

---

## 📊 Visual Results

### Performance Summary by Data Type

| Data Type     | Avg. NRMS | Avg. AE |
|---------------|-----------|---------|
| Numerical     | 0.412     |   —     |
| Categorical   |    —      | 0.532   |
| Mixed         | 0.468     | 0.489   |

> *Performance decreases with higher missing percentages and larger datasets due to memory and neighbor selection limitations.*
