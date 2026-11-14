"""
Iris Flower Classification using K-Nearest Neighbors (KNN)
-----------------------------------------------------------
Steps:
  1. Load and explore the dataset
  2. Split into train / test sets
  3. Scale features (KNN is distance-based, so scaling matters)
  4. Choose the best k using cross-validation
  5. Train and evaluate the final model
  6. Visualise results and predict a new flower

Run:  python iris_knn.py
Requires: scikit-learn, pandas, matplotlib, seaborn
"""

import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up on screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------- 1. Load data
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("=== Dataset overview ===")
print(df.head(), "\n")
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("\nClass distribution:\n", df["species"].value_counts(), "\n")

# Pairplot to see how separable the classes are
sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------- 2. Train/test split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# ------------------------------------------- 3 & 4. Pick best k with 5-fold CV
# A pipeline keeps scaling inside each CV fold, avoiding data leakage.
k_values = range(1, 26)
cv_scores = []
for k in k_values:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[int(np.argmax(cv_scores))]
print(f"Best k from cross-validation: {best_k} "
      f"(CV accuracy = {max(cv_scores):.4f})\n")

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), cv_scores, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Cross-validated accuracy")
plt.title("Choosing k for KNN")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("iris_k_selection.png", dpi=120, bbox_inches="tight")
plt.close()

# -------------------------------------------------- 5. Train final model & test
final_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("=== Test set results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("iris_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------- 6. Decision boundary (2 features, for visuals)
feat_idx = [2, 3]  # petal length, petal width: the most discriminative pair
X2_train = X_train.iloc[:, feat_idx].values
X2_test = X_test.iloc[:, feat_idx].values
model_2d = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
model_2d.fit(X2_train, y_train)

x_min, x_max = X.iloc[:, 2].min() - 0.5, X.iloc[:, 2].max() + 0.5
y_min, y_max = X.iloc[:, 3].min() - 0.5, X.iloc[:, 3].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
zz = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="viridis")
scatter = plt.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test,
                      cmap="viridis", edgecolor="k")
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title(f"KNN decision boundary (k={best_k}), test points shown")
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))
plt.savefig("iris_decision_boundary.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------ Predict a new flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
pred = final_model.predict(new_flower)[0]
proba = final_model.predict_proba(new_flower)[0]
print("=== New sample prediction ===")
print(new_flower.to_string(index=False))
print("Predicted species:", iris.target_names[pred])
print("Class probabilities:",
      {str(n): round(float(p), 3) for n, p in zip(iris.target_names, proba)})
print("\nSaved plots: iris_pairplot.png, iris_k_selection.png, "
      "iris_confusion_matrix.png, iris_decision_boundary.png")
