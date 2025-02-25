import json
import pickle,pandas as pd
from sklearn.metrics import accuracy_score, f1_score,precision_score,log_loss, confusion_matrix
# Load test data
test = pd.read_csv(r"D:\MLOPS ASS=1\test.csv, index=False")
X_test = test.drop(columns=["left"])  # Drops the "left" column to keep features
y_test = test["left"]  # Target variable
results = {}
for model_name, model_file in [("RandomForest", r"D:\MLOPS ASS=1\random_forest.pkl"), ("LogReg", r"D:\MLOPS ASS=1\logistic_regression.pkl")]:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[model_name] = {"accuracy": accuracy, 
                           "f1_score": f1, 
                           "Log Loss":loss,
                           "precision":precision,
                           "Confusion Matrix": cm.tolist() }
# Save results
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)
print("✅ Model evaluation complete. Results saved.")
