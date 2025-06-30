# Task 5 - Decision Trees and Random Forests using Heart Disease Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
# 1. Load Dataset
df = pd.read_csv("heart.csv")
print("Dataset Shape:", df.shape)
print(df.head())
# 2. Split features and target
X = df.drop("target", axis=1)
y = df["target"]
# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
# Evaluate Decision Tree
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
# Visualize Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
# 5. Control Overfitting
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print("Pruned Decision Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))
# 6. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# 7. Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
sns.barplot(x=[X.columns[i] for i in indices], y=importances[indices])
plt.title("Feature Importances - Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 8. Cross-Validation
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print("Cross-validation Accuracy (Decision Tree):", np.mean(cv_scores_dt))
print("Cross-validation Accuracy (Random Forest):", np.mean(cv_scores_rf))
# 9. Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))