# Seller Churn Prediction (E-commerce)

Predicting high-risk sellers using machine learning to enable targeted retention strategies and reduce revenue loss.

---

## 📊 Business Problem
Seller churn impacts platform performance beyond just seller count—it is strongly linked to weaker operational performance and reduced revenue contribution.

Churned sellers exhibit:
- **Lower fulfillment performance**, including longer dispatch times and significantly higher late delivery rates  
- **Substantially lower revenue and order volume**, contributing only a small share of total platform performance  

While churned sellers may have higher average order value, they transact less frequently, limiting their overall impact. These patterns suggest that early operational inefficiencies—particularly in fulfillment—are key leading indicators of churn.

This project aims to identify high-risk sellers early and enable targeted interventions to improve retention, operational performance, and overall platform revenue.

---

## 🎯 Objectives
- Identify key differences between churned and active sellers  
- Analyze drivers of seller churn  
- Build a predictive model to detect high-risk sellers  

---

## 🛠️ Approach
- Performed **Exploratory Data Analysis (EDA)** and feature engineering  
- Trained and evaluated multiple classification models:
  - Decision Tree
  - SVM
  - KNN
  - XGBoost  
- Optimized model using **F2-score** to prioritize recall of churners  
- Applied **SHAP (Explainable AI)** to identify key churn drivers  

---

## 📈 Results
- Achieved **~8x improvement in churn detection efficiency** compared to no-model baseline  
- Identified key drivers:
  - Selling frequency  
  - Revenue trends  
  - Delivery performance  
- Reduced false negatives significantly, improving early detection of at-risk sellers  

---

## 💡 Business Impact
- Enables **targeted retention strategies**  
- Reduces wasted effort on low-risk sellers  
- Supports more efficient resource allocation  

---

## 📊 Dashboard & Monitoring
Developed a monitoring dashboard to track seller behavior and provide early warning signals for churn risk.

---

## 🧰 Tech Stack
- Python (Pandas, NumPy, Scikit-learn)  
- Decision Tree, XGBoost, SVM, KNN
- SHAP  
- Matplotlib / Seaborn  

---

## 👥 Authors
- Naufal  
- Margaretha Kwok  
