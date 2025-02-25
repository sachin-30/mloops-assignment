# process.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load dataset
df = pd.read_csv(r"D:\MLOPS ASS=1\HR_comma_sep.csv")
# Handle missing values
df.dropna(inplace=True)
# Encoding categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
label_encoders[col] = le
# Split dataset into features and target
X = df.drop(columns=['left'])
y = df['left']
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Save preprocessed data
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['left'] = y_train.values
train_df.to_csv(r"D:\MLOPS ASS=1\train.csv, index=False")
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['left'] = y_test.values
test_df.to_csv(r"D:\MLOPS ASS=1\test.csv, index=False")
