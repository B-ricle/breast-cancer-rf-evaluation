from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)
print(f"Dataset: Wisconsin Breast Cancer ({len(X)} rows)")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_model = RandomForestClassifier(
n_estimators=100,
max_depth=5,
random_state=42,
n_jobs=-1
)
scoring = {
"accuracy": "accuracy",
"precision": "precision",
"recall": "recall",
"f1": "f1",
"roc_auc": "roc_auc"
}
cv_results = cross_validate(
cv_model, X_train, y_train,
cv=cv, scoring=scoring,
return_train_score=False
)

cv_scores = pd.DataFrame({
    "Fold": range(1, 6),
    "Accuracy": cv_results["test_accuracy"],
    "Precision": cv_results["test_precision"],
    "Recall": cv_results["test_recall"],
    "F1": cv_results["test_f1"],
    "ROC-AUC": cv_results["test_roc_auc"]
})

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_baseline = baseline.predict(X_val)

print(accuracy_score(y_val, y_baseline))
print(recall_score(y_val, y_baseline))
print(f1_score(y_val, y_baseline))
print(precision_score(y_val, y_baseline, zero_division=0))
print(confusion_matrix(y_val, y_baseline))

model = RandomForestClassifier(
n_estimators=100,
max_depth=None,
random_state=42,
n_jobs=-1
)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]
model_A = RandomForestClassifier(
    n_estimators=50,
    max_depth = 3,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
model_B = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
model_C = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight=None, 
    random_state=42,
    n_jobs=-1
)
model_D = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

modelA_results = cross_validate(
    model_A, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False
)

modelB_results = cross_validate(
    model_B, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False
)
modelC_results = cross_validate(
    model_C, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False
)
modelD_results = cross_validate(
    model_D, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False
)

mean_rows = []
std_rows = []

result = {
    "A": modelA_results,
    "B": modelB_results,
    "C": modelC_results,
    "D": modelD_results
}

for model_name, model_result in result.items():
    mean_row = {
        "Model": model_name,
        "Mean Accuracy": model_result["test_accuracy"].mean(),
        "Mean Precision": model_result["test_precision"].mean(),
        "Mean Recall": model_result["test_recall"].mean(),
        "Mean F1": model_result["test_f1"].mean(),
        "Mean ROC-AUC": model_result["test_roc_auc"].mean()
    }

    std_row = {
        "Model": model_name,
        "Std Accuracy": model_result["test_accuracy"].std(),
        "Std Precision": model_result["test_precision"].std(),
        "Std Recall": model_result["test_recall"].std(),
        "Std F1": model_result["test_f1"].std(),
        "Std ROC-AUC": model_result["test_roc_auc"].std()
    }
    mean_rows.append(mean_row)
    std_rows.append(std_row)

    mean_results = pd.DataFrame(mean_rows)
    std_results = pd.DataFrame(std_rows)


# Model D selected 

model_D.fit(X_train, y_train)
val_pred = model_D.predict(X_val)
val_prob = model_D.predict_proba(X_val)[:, 1]

print(accuracy_score(y_val, val_pred))
print(recall_score(y_val, val_pred))
print(f1_score(y_val, val_pred))
print(precision_score(y_val, val_pred))
print(confusion_matrix(y_val, val_pred))
print(roc_auc_score(y_val, val_prob))

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_final, y_train_final)

y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 3))
print("Precision:", round(precision_score(y_test, y_test_pred), 3))
print("Recall:", round(recall_score(y_test, y_test_pred), 3))
print("F1:", round(f1_score(y_test, y_test_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_prob), 3))

print("Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

importance_df = pd.DataFrame({
"Feature": X.columns,
"Importance": final_model.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.head(10).to_string(index=False))

top_10 = importance_df.head(10)

plt.figure(figsize=(8, 6))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.show()