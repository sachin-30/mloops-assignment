#train.py
import pickle,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Load preprocessed data
train = pd.read_csv(r"D:\MLOPS ASS=1\test.csv, index=False")
X_train = train.drop(columns=["left"])  # Drops the "left" column to keep features
y_train = train["left"]  # Target variable
# Define ML Pipelines
pipeline_rf = Pipeline([("model", RandomForestClassifier(n_estimators=100))])
pipeline_lr = Pipeline([("model", LogisticRegression())])
# Train Models
pipeline_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
# Save Models
with open(r"D:\MLOPS ASS=1\random_forest.pkl", "wb") as f:
    pickle.dump(pipeline_rf, f)

with open(r"D:\MLOPS ASS=1\logistic_regression.pkl", "wb") as f:
    pickle.dump(pipeline_lr, f)
print("✅ Models trained and saved.")


