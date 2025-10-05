import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

mnb = MultinomialNB()
bnb = BernoulliNB()
gnb = GaussianNB()

le = LabelEncoder()
df = pd.read_csv('s.csv')
df.drop_duplicates(subset=['customerID'], inplace=True)
df.drop('customerID', axis=1, inplace=True)  # 💡 Important drop

scalar = MinMaxScaler()
model = LogisticRegression()

for_churn = df[['SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]

# One-hot encode categorical columns
for_churn = pd.get_dummies(for_churn, columns=[
    'PaymentMethod','MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract'
], drop_first=False)

# Clean True/False
for_churn.replace({False: 0, True: 1}, inplace=True)

# Label encode binary columns
for_churn['Partner'] = le.fit_transform(for_churn['Partner'])
for_churn['Dependents'] = le.fit_transform(for_churn['Dependents'])
for_churn['PhoneService'] = le.fit_transform(for_churn['PhoneService'])
for_churn['PaperlessBilling'] = le.fit_transform(for_churn['PaperlessBilling'])
for_churn['Churn'] = le.fit_transform(for_churn['Churn'])

# Convert TotalCharges to numeric
for_churn['TotalCharges'] = pd.to_numeric(for_churn['TotalCharges'], errors='coerce')
for_churn.dropna(inplace=True)

# Define X and y
X = for_churn.drop('Churn', axis=1)
y = for_churn['Churn']  # ✅ FIXED: y ko scalar se transform mat karo

# Scale features
Scaled_X = scalar.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(Scaled_X, y, test_size=0.2, random_state=42)

# Logistic Regression
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Logistic Regression")
print(f"Accuracy : {accuracy_score(y_test, prediction)}")
print(f"Precision : {precision_score(y_test, prediction)}")

# MNB
mnb.fit(X_train, y_train)
mnbprediction = mnb.predict(X_test)  # ✅ FIXED
print("MNB")
print(f"Accuracy : {accuracy_score(y_test, mnbprediction)}")
print(f"Precision : {precision_score(y_test, mnbprediction)}")

# BNB
bnb.fit(X_train, y_train)
bnbprediction = bnb.predict(X_test)  # ✅ FIXED
print("BNB")
print(f"Accuracy : {accuracy_score(y_test, bnbprediction)}")
print(f"Precision : {precision_score(y_test, bnbprediction)}")

# GNB
gnb.fit(X_train, y_train)
gnbprediction = gnb.predict(X_test)  # ✅ FIXED
print("GNB")
print(f"Accuracy : {accuracy_score(y_test, gnbprediction)}")
print(f"Precision : {precision_score(y_test, gnbprediction)}")
