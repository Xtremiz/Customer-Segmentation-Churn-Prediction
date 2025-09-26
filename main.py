import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

le= LabelEncoder()
df = pd.read_csv('s.csv')
df.drop_duplicates(subset=['customerID'],inplace=True)

for_churn = pd.get_dummies(df,columns=['Contract'], drop_first=False)

for_churn = df[['SeniorCitizen', 'Partner','tenure','Dependents','InternetService','StreamingTV', 'StreamingMovies','Contract','PaymentMethod', 'MonthlyCharges','TotalCharges','Churn']]
for_churn = pd.get_dummies(for_churn,columns=['InternetService','StreamingTV', 'StreamingMovies','Contract','PaymentMethod'], drop_first=False)
for_churn.replace({True:1,False:0},inplace=True)
print(for_churn.columns)

"""for i in for_churn.columns:
    print(f"--- {i} ------")
    print(for_churn[i].value_counts())"""
