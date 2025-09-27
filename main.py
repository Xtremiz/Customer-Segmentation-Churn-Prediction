import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score


le= LabelEncoder()
df = pd.read_csv('s.csv')
df.drop_duplicates(subset=['customerID'],inplace=True)

for_churn = pd.get_dummies(df,columns=['Contract'], drop_first=False)

for_churn = df[['gender','SeniorCitizen', 'Partner','tenure','Dependents','InternetService','StreamingTV', 'StreamingMovies','Contract','PaymentMethod', 'MonthlyCharges','TotalCharges','Churn']]
for_churn = pd.get_dummies(for_churn,columns=['InternetService','StreamingTV', 'StreamingMovies','Contract','PaymentMethod'], drop_first=False)
for_churn.replace({True:1,False:0},inplace=True)
for_churn['Partner']=le.fit_transform(for_churn['Partner'])
for_churn['Dependents']=le.fit_transform(for_churn['Dependents'])
for_churn['Churn']=le.fit_transform(for_churn['Churn'])
for_churn['gender']=le.fit_transform(for_churn['gender'])
for_churn['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

for_churn.dropna(inplace=True)

model = LogisticRegression()

X = for_churn[['gender','SeniorCitizen', 'Partner', 'tenure', 'Dependents', 'MonthlyCharges',
       'TotalCharges','InternetService_DSL','InternetService_Fiber optic', 'InternetService_No', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']]
y=  for_churn['Churn']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print(f"accuracy :{accuracy_score(y_test,prediction)}")
print(f"prescision :{precision_score(y_test,prediction)}")

print(df.columns)