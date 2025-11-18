import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier   




# Load dataset
df = pd.read_csv("heart_disease_uci.csv")


#print(df.head(5))

# Check missing values in percentage
#print("Missing values (%):")
#print(df.isnull().sum() * 100)

# ---- Fill numerical columns with mean ----
df["trestbps"] = df["trestbps"].fillna(df["trestbps"].mean())
df["chol"]     = df["chol"].fillna(df["chol"].mean())
df["thalch"]   = df["thalch"].fillna(df["thalch"].mean())
df["oldpeak"]  = df["oldpeak"].fillna(df["oldpeak"].mean())

# ---- Fill numerical column with median ----
df["ca"] = df["ca"].fillna(df["ca"].median())

# ---- Fill categorical/boolean column with mode ----
df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0]).astype(int)
df["exang"] = df["exang"].fillna(df["exang"].mode()[0]).astype(int)

#df["restecg"] = df["restecg"].fillna(df["restecg"].mode()[0]).astype(int)




df_encoded = pd.get_dummies(df,columns=['slope'])
df_encoded = pd.get_dummies(df_encoded,columns=['thal'])
df_encoded = pd.get_dummies(df_encoded,columns=['cp'])
df_encoded = pd.get_dummies(df_encoded,columns=['dataset'])

df_label = df_encoded.copy()

le = LabelEncoder()

df_label['restecg'] = le.fit_transform(df_label['restecg'])
df_label['sex'] = le.fit_transform(df_label['sex'])

'''
# Check again after filling
print("\nMissing values after cleaning:")
print(df_label.isnull().sum())
'''
print("Duplicate values sum : ",df_label.duplicated().sum())

df_label['num'] = (df_label['num'] > 0).astype(int) #making num as the binary as it is y 


nums_scale = ['age','trestbps','chol','thalch','oldpeak','ca']

scalar = StandardScaler()

df_label[nums_scale] = scalar.fit_transform(df_label[nums_scale] )

#print(df_label.head())

#now data is ready to train 

X = df_label.drop("num",axis=1)
y = df_label['num']

X_train , X_test , y_train , y_test =train_test_split(X,y , test_size=0.2,random_state=42,stratify=y)



X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)


log_reg = LogisticRegression(max_iter=1000)  
log_reg.fit(X_train, y_train)  
y_pred_lr = log_reg.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)  
y_pred_knn = knn.predict(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# Logistic Regression
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# KNN
print("\nKNN Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest" : RandomForestClassifier( n_estimators=100,criterion="gini", max_depth=None,   random_state=42 )    
        
  
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
   # print("F1 Score:", f1_score(y_test, y_pred))
   # print("Classification Report:\n", classification_report(y_test, y_pred))
   # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    trained_models[name] = model   # store trained model


print("Tell me following things :\n ")
# Collect user inputs
id = int(input("Enter your ID: "))
age = float(input("Enter your age: "))
sex = float(input("Enter your sex (0 = female, 1 = male): "))
cp = float(input("Enter chest pain type (0-3): "))
trestbps = float(input("Enter resting blood pressure: "))
chol = float(input("Enter serum cholesterol: "))
fbs = float(input("Enter fasting blood sugar > 120 mg/dl (1 = true, 0 = false): "))
restecg = float(input("Enter resting ECG results (0-2): "))
thalach = float(input("Enter maximum heart rate achieved: "))
exang = float(input("Enter exercise induced angina (1 = yes, 0 = no): "))
oldpeak = float(input("Enter ST depression induced by exercise: "))
slope = float(input("Enter slope of the ST segment (0-2): "))
ca = float(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
thal = float(input("Enter thal (1 = normal, 2 = fixed defect, 3 = reversible defect): "))





# Create a dictionary for user input
user_input = {
    'age': age,
    'sex': sex,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'ca': ca,
    # one-hot encode cp (0–3)
    'cp_0': 1 if cp == 0 else 0,
    'cp_1': 1 if cp == 1 else 0,
    'cp_2': 1 if cp == 2 else 0,
    'cp_3': 1 if cp == 3 else 0,
    # one-hot encode slope (0–2)
    'slope_0': 1 if slope == 0 else 0,
    'slope_1': 1 if slope == 1 else 0,
    'slope_2': 1 if slope == 2 else 0,
    # one-hot encode thal (1–3)
    'thal_1': 1 if thal == 1 else 0,
    'thal_2': 1 if thal == 2 else 0,
    'thal_3': 1 if thal == 3 else 0,
    # you also encoded "dataset" → assume possible values are 0,1,2,3
    'dataset_0': 0, 'dataset_1': 0, 'dataset_2': 0, 'dataset_3': 0
}

# Convert dictionary into DataFrame
user_df = pd.DataFrame([user_input])


#Add any missing columns as 0
for col in X.columns:
    if col not in user_df.columns:
        user_df[col] = 0


# Reorder columns exactly like training data
user_df = user_df[X.columns]

# Scale numeric columns
user_df[nums_scale] = scalar.transform(user_df[nums_scale])


"""
for name, model in trained_models.items():
    prediction = model.predict(user_df)[0]
    print(f"\n{name} Prediction:")
    if prediction == 1:
        print("⚠️ Heart Disease")
    else:
        print("✅ No Heart Disease")

"""
final_preds = []   # to store predictions from each model

for name, model in trained_models.items():
    prediction = model.predict(user_df)[0]
    final_preds.append(prediction)

    print(f"\n{name} Prediction:")
    if prediction == 1:
        print("⚠️ Heart Disease")
    else:
       print("✅ No Heart Disease")

# --- Majority Voting ---
votes = sum(final_preds)
if votes > len(final_preds) // 2:
    final_prediction = 1
else:
    final_prediction = 0

print("\n🟢 Final Majority Vote Result:")
if final_prediction == 1:
    print("⚠️ Ensemble Prediction: Heart Disease")
else:
    print("✅ Ensemble Prediction: No Heart Disease")



'''
# Predict
prediction = model.predict(user_df)[0]

if prediction == 1:
    print("⚠️ The model predicts: Heart Disease")
else:
    print("✅ The model predicts: No Heart Disease")

'''