# 🚗 Car Price Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Toolkit-F7931E.svg)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0.svg)
![Machine Learning](https://img.shields.io/badge/Model-Linear%20Regression-brightgreen)

Predicting the selling price of cars based on various features using **Linear Regression**, **OneHotEncoder**, **Pipelines**, and thorough **EDA**.  
This project also includes **model deployment using Pickle**, making it possible to integrate the trained model into applications or APIs.

---

##  Project Overview

This machine learning project predicts used car prices by analyzing key features such as brand, fuel type, transmission, engine power, and mileage.

It covers:

-  Comprehensive EDA  
-  Data preprocessing  
-  OneHotEncoding of categorical features  
-  Pipelines for end-to-end processing  
-  Linear Regression Model  
-  R² Score evaluation  
-  Deployment using Pickle (`.pkl`)  

---

#    Workflow 

## 1) Importing Libraries
All essential libraries were imported for cleaning, exploration, visualization, preprocessing, and model building:

- `pandas`, `numpy` → Data manipulation  
- `matplotlib`, `seaborn` → Visualizations and EDA  
- `scikit-learn` → OneHotEncoder, Pipeline, Linear Regression, model evaluation  

---

## 2) Data Loading & Cleaning
- Dataset used: **quikr.csv**  
- Removed irrelevant or redundant columns  
- Treated missing values using:
- Converted categorical features using:
  - `OneHotEncoder` (inside ML Pipeline)

---

## 3) Exploratory Data Analysis (EDA)
Performed visual and statistical analysis to understand data patterns:

- Boxplots → outlier detection  
- **relplot (Seaborn Relational Plot)** → multi-dimensional relationships  
  - Example: `sns.relplot(x="mileage", y="price", hue="fuel_type")`
- **swarmplot** → distribution of categorical vs numeric features  
  - Example: `sns.swarmplot(x="fuel_type", y="price")`

These insights guided preprocessing and model selection.

---

## 4) Data Pre-Processing
Preprocessing was done using a **Scikit-Learn Pipeline**, which included:

- OneHotEncoding of categorical columns  
- Numeric feature handling  
- Train–Test Split (80% training, 20% testing)

This ensures consistency and avoids manual processing during prediction.

---

## 5) Model Training
- **Algorithm Used:** `LinearRegression`
- Model was trained inside a Pipeline that included:
  - OneHotEncoding  
  - Linear Regression model  

This ensures smooth and automated processing from raw input → prediction.

---

## 6) Model Evaluation
Model performance measured using:

- **R² Score**  

These metrics helped determine prediction accuracy and reliability.

---

## 7) Model Saving (Pickle File)
The final trained model (including encoding + preprocessing) was saved as a **Pickle (.pkl)** file.

This allows the model to be loaded instantly without retraining.

---

# 📈 Model Used — Linear Regression

Linear Regression is ideal for predicting **continuous values** like car prices.  
It identifies relationships between features such as:

- Engine power  
- Mileage  
- Car brand  
- Fuel type  
- Transmission  
- Car age  

The entire preprocessing pipeline + model is saved inside the `.pkl` file.

---

# 📦 Using the Pickle Model (Car Price Prediction)

```python
import pickle

# Save the trained model pipeline
pickle.dump(model_pipeline, open('LinearRegressionModel.pkl', 'wb'))

# Load the model pipeline for predictions
loaded_model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Example prediction input
# Replace values with actual feature inputs after preprocessing
sample_input = [[year, mileage, engine, power, transmission, fuel_type, brand]]

prediction = loaded_model.predict(sample_input)
print("Predicted Price:", prediction)

```
## Significance of the Pickle File

The **model.pkl** file stores the trained Random Forest model in binary format.

**1) Model Reusability**: You don’t need to retrain the model every time you run predictions.

**2) Deployment Ready**: Can be easily integrated into web apps (Flask, Streamlit, etc.) for live predictions.

**3) Efficiency**: Reduces computational load and saves time.

**4) Consistency**: Ensures the same trained model is used across all environments.

 **model.pkl** acts as the bridge between model training and deployment, enabling real-time car price predictions.
   
![image alt](https://github.com/sanjanabh127/Car-Price-Prediction/blob/a5efd5da748d64d96b45b564693afd19c248b7b7/Screenshot%202025-12-02%20144425.png)

![image alt](https://github.com/sanjanabh127/Car-Price-Prediction/blob/1a611661ecb3e974e678a82acd7689433652966a/Screenshot%202025-12-02%20144437.png)
