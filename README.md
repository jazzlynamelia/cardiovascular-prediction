# Cardiovascular Disease Prediction with Apache Spark ML

This repository contains a **Big Data pipeline** project developed for the *Big Data Processing* course. The goal is to predict cardiovascular disease using **Apache Spark ML** and machine learning on large-scale medical data.

---

## Overview
Cardiovascular disease (CVD) accounts for ~13% of global deaths. Early detection is challenging due to complex and large-scale medical data.  
This project demonstrates how **Apache Spark** can be leveraged for scalable data preprocessing, feature engineering, and model training to support **early risk detection** and preventive healthcare.

Three models were implemented:
- **Logistic Regression**
- **Random Forest**
- **Gradient-Boosted Tree (GBTClassifier)**

**Best Result**: Gradient-Boosted Tree with **89.73% accuracy** and **90.31% precision**.

---

## Workflow
<p align="center">
  <img src="assets/workflow.png" alt="Workflow Diagram" width="900"/>
</p>

---

## Methodology
1. **Data Collection**  
   - Dataset: [Cardiovascular Disease Dataset (Kaggle)](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
   - 70,000 patient records (demographics, lifestyle, medical features).

2. **Data Cleaning**  
   - Removed irrelevant columns (`id`), handled missing values, duplicates, outliers (IQR).  
   - Validated categorical values, standardized formats.  
   <p align="center">
     <img src="assets/histograms.png" alt="Histograms of Features" width="600"/>
   </p>
   <p align="center">
     <img src="assets/boxplots_before.png" alt="Boxplots Before Cleaning" width="600"/>
     <img src="assets/boxplots_after.png" alt="Boxplots After Cleaning" width="600"/>
   </p>

3. **Feature Engineering & Selection**  
   - Created new variables: BMI, blood pressure categories, age groups, pressure ratios, cholesterol-BMI interaction, lifestyle impact.  
   - Dropped redundant/low-importance features.  

   <p align="center">
   <table>
     <tr>
       <td><img src="assets/distribution1.png" alt="Distribution Feature 1" width="300"/></td>
       <td><img src="assets/distribution2.png" alt="Distribution Feature 2" width="300"/></td>
     </tr>
     <tr>
       <td><img src="assets/distribution3.png" alt="Distribution Feature 3" width="300"/></td>
       <td><img src="assets/distribution4.png" alt="Distribution Feature 4" width="300"/></td>
     </tr>
   </table>
   </p>

   <p align="center">
     <img src="assets/correlation_matrix.png" alt="Correlation Matrix" width="600"/>
   </p>

4. **Preprocessing**  
   - One-Hot Encoding for categorical variables.  
   - StandardScaler for continuous variables.  
   - Train/test split (80:20).  

5. **Model Training & Evaluation**  
   - Models: Logistic Regression, Random Forest, Gradient-Boosted Tree.  
   - Hyperparameter tuning with 2-fold Cross Validation.  
   - Metrics: Accuracy, Precision, Recall, F1-score.  

   <p align="center">
     <img src="assets/cm_lr.png" alt="Confusion Matrix - Logistic Regression" width="250"/>
     <img src="assets/cm_rf.png" alt="Confusion Matrix - Random Forest" width="250"/>
     <img src="assets/cm_gbt.png" alt="Confusion Matrix - Gradient-Boosted Tree" width="250"/>
   </p>

   <p align="center">
     <img src="assets/model_comparison.png" alt="Model Performance Comparison" width="600"/>
   </p>

---

## Results
- **Gradient-Boosted Tree** → Accuracy **89.73%**, Precision **90.31%**.  
- **Random Forest** → Accuracy **89.52%**, Precision **90.30%**.  
- **Logistic Regression** → Accuracy **89.68%**, Precision **80.46%**.  

All models performed competitively, but GBT showed the most stable and consistent results.

---

## Tools & Libraries
- **Apache Spark MLlib**
- **Python (Google Colab)**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn** (for evaluation metrics)

---

## How to Run

Since this project was developed in **Google Colab**, you can run it without setting up a local environment:

1. Open [Google Colab](https://colab.research.google.com/)  
2. Click **File > Open notebook > GitHub**  
3. Enter this repository URL: [https://github.com/jazzlynamelia/cardiovascular-prediction/](https://github.com/jazzlynamelia/cardiovascular-prediction/)
4. Select the notebook you want to run (`cardiovascular_prediction.ipynb`)  
5. Run the notebook step by step
