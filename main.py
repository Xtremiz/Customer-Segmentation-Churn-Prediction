import pandas as pd
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
df = pd.read_csv('s.csv')
df.drop_duplicates(subset=['customerID'],inplace=True)

print(df.info())