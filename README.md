Heart-Disease-Prediction-
This project predicts whether a person has heart disease based on clinical features.  
It uses multiple machine learning models including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**, and then performs **majority voting** to give a final ensemble prediction.

Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can save lives.  
This project uses the **UCI Heart Disease Dataset** to train models that can classify a patient as:

0 → No Heart Disease
1 → Heart Disease

The script:
Cleans missing values  
Encodes categorical features  
Applies feature scaling  
Trains multiple ML models  
Takes user input  
Predicts heart disease using an ensemble method  

Technologies Used
Python  
NumPy  
Pandas  
Matplotlib  
Seaborn  
Scikit-learn (sklearn)

Machine Learning Models Used
1. Logistic Regression
2. KNN (K-Nearest Neighbors)
3. Random Forest Classifier
4. Ensemble (Majority Voting)

Each model makes its own prediction, and the ensemble combines them for a more reliable result.


