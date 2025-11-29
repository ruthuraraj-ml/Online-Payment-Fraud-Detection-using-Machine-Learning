# 💳 Online Payment Fraud Detection using Machine Learning

This project builds machine learning models to detect fraudulent online payment transactions in a highly imbalanced real-world dataset.  
Techniques include EDA, class imbalance handling, feature scaling, baseline modeling, ensemble learning, and multi-metric evaluation.

The final model (XGBoost) demonstrates excellent fraud detection capability with strong Recall, Precision, ROC–AUC, and PR–AUC performance.

---

## 📌 Project Objective
The goal is to detect **fraudulent transactions (1) vs legitimate transactions (0)** using anonymized numerical features.

Key objectives:

- Perform EDA to understand class imbalance and feature patterns  
- Apply imbalance-aware preprocessing and scaling  
- Train Logistic Regression, Random Forest, and XGBoost  
- Use Recall, Precision, F1, ROC–AUC, PR–AUC (instead of accuracy)  
- Compare models and determine the best fraud detection strategy  
- Identify important features influencing fraud prediction  

---

## 📁 Dataset Overview

- **Total records:** 56,962  
- **Fraudulent transactions:** 88  
- **Legitimate transactions:** 56,874  
- **Fraud ratio:** ~0.15% (extreme imbalance)  
- **Features:** 30 anonymized PCA-like numerical variables + Amount + Time  
- **Target:** `Class` (0 = non-fraud, 1 = fraud)

### Data Characteristics
- Features mostly standardized  
- Only `Amount` required scaling  
- PCA-like features → approximately independent  
- Outliers retained (they represent real fraudulent behavior)

---

## 🔍 Exploratory Data Analysis — Key Insights

### **Class Distribution**
- Severe imbalance (≈0.15% fraud)  
- Accuracy alone is misleading → imbalance-aware metrics required

### **Feature Insights**
- Amount shows large variance  
- Time feature optional (temporal pattern not strong)  
- PCA-like features are clean and usable  
- No strong multicollinearity  

### **Outliers**
- Most extreme values reflect actual fraud cases → kept intentionally  

---

## 🧱 Data Preprocessing

### **Imbalance Handling**
- *Logistic Regression:* `class_weight="balanced"`  
- *Random Forest, XGBoost:* handled naturally through tree splits  
- ROC–AUC and PR–AUC used for meaningful evaluation

### **Scaling**
- StandardScaler applied only to `Amount`  
- Prevents LR instability; tree models unaffected  

### **Train–Test Split**
- 80% train / 20% test  
- Stratified sampling to maintain class ratio  

---

## 🧠 Models Implemented

### **1️⃣ Logistic Regression (Baseline)**
- Interpretable  
- Class-weight balancing used  

**Performance:**

| Metric | Fraud Class |
|--------|-------------|
| Precision | 0.06 |
| Recall | 0.91 |
| F1-score | 0.12 |
| ROC–AUC | 0.99 |
| PR–AUC | 0.69 |

**Summary:**  
Excellent recall (catches frauds), very poor precision (many false alarms).  
Useful for first-layer screening.

---

### **2️⃣ Random Forest Classifier**
- Robust ensemble  
- Handles imbalance well  
- No scaling required  

**Performance:**

| Metric | Fraud Class |
|--------|-------------|
| Precision | 0.93 |
| Recall | 0.77 |
| F1-score | 0.84 |
| ROC–AUC | 0.83 |
| PR–AUC | 0.81 |

**Summary:**  
High precision (few false alarms), moderate recall.  
Good for secondary verification.

---

### **3️⃣ XGBoost Classifier (Best Model)**
- Superior handling of rare events  
- Gradient boosting improves discrimination  

**Performance:**

| Metric | Fraud Class |
|--------|-------------|
| Precision | 0.90 |
| Recall | 0.83 |
| F1-score | 0.86 |
| ROC–AUC | 0.97 |
| PR–AUC | 0.82 |

**Summary:**  
Best balance of Recall + Precision + AUC metrics.  
Ideal for real-world detection and deployment.

---

## 🎯 Model Comparison Summary

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC–AUC | PR–AUC |
|-------|--------------------|-----------------|-------------|----------|--------|
| Logistic Regression | 0.06 | **0.91** | 0.12 | **0.99** | 0.69 |
| Random Forest | **0.93** | 0.77 | 0.84 | 0.83 | 0.81 |
| XGBoost | 0.90 | **0.83** | **0.86** | 0.97 | **0.82** |

**Winner:**  
👉 **XGBoost** — combines strong recall with high precision and top AUC values.

---

## 🔐 Real-World Insights

- Fraud detection must minimize **false negatives** (missed frauds)  
- Threshold tuning is essential (business-dependent)  
- Precision–Recall curves give more useful insights than accuracy  
- Ensemble models capture non-linear transaction patterns  
- PCA-based anonymized features still reveal fraud indicators

---

## 📦 Key Results Summary

| Item | Result |
|------|--------|
| Best Model | **XGBoost** |
| Accuracy | 0.9996 |
| Recall (Fraud) | 0.83 |
| Precision (Fraud) | 0.90 |
| F1-score (Fraud) | 0.86 |
| ROC–AUC | 0.97 |
| PR–AUC | 0.82 |
| Dataset Size | 56,962 records |

---

## 🧱 Tech Stack
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Jupyter / Google Colab  
