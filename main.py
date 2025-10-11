import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score,silhouette_score
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture

le = LabelEncoder()
df = pd.read_csv('s.csv')
df.drop_duplicates(subset=['customerID'], inplace=True)
df.drop('customerID', axis=1, inplace=True)  # 💡 Important drop

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

for_churn['TotalCharges'] = pd.to_numeric(for_churn['TotalCharges'], errors='coerce')
for_churn.dropna(inplace=True)

X = for_churn.drop('Churn', axis=1)
y = for_churn['Churn']  

model = RandomForestClassifier(class_weight="balanced",random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # or use pos_label='Yes' if labels are binary

print("Final Evaluation Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")

#now applying the unsupervised learning
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
X_pca['Churn'] = for_churn['Churn'].values


kmeans = KMeans(n_clusters=2,random_state=42)
kmeansprediction = kmeans.fit_predict(X_pca)
X_pca['Cluster'] = kmeansprediction
kmeanssscore = silhouette_score(X_pca[['PCA1', 'PCA2']],kmeansprediction)
print(f"Kmeans score is {kmeanssscore:.4f}")
