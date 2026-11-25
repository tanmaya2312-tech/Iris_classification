```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42


# 1. Load the Iris Dataset

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)

print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print()
print(df["species"].value_counts())

df.head()


# 2. Descriptive Statistics

df.drop(columns="target").describe()


# 3. Data Visualization - Pair Plot

sns.pairplot(
    df.drop(columns="target"),
    hue="species",
    corner=True
)
plt.show()


# 4. Train-Test Split

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train size:", len(X_train), "| Test size:", len(X_test))


# 5. Finding the Best K using 5-Fold Cross-Validation

k_values = range(1, 26)
cv_scores = []

for k in k_values:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    cv_scores.append(
        cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        ).mean()
    )

best_k = k_values[int(np.argmax(cv_scores))]

print(
    f"Best k = {best_k} "
    f"(CV accuracy = {max(cv_scores):.4f})"
)


# 6. Visualizing K Selection

plt.figure(figsize=(7, 4))

plt.plot(list(k_values), cv_scores, marker="o")

plt.axvline(
    best_k,
    color="red",
    linestyle="--",
    label=f"best k = {best_k}"
)

plt.xlabel("k (number of neighbours)")
plt.ylabel("Cross-validated accuracy")
plt.title("Choosing k for KNN")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# 7. Final Model Training

final_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)


# 8. Model Evaluation

print(
    f"Test accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# 9. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    cm,
    display_labels=iris.target_names
).plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()


# 10. KNN Decision Boundary Visualization

feat_idx = [2, 3]

X2_train = X_train.iloc[:, feat_idx].values
X2_test = X_test.iloc[:, feat_idx].values

model_2d = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)

model_2d.fit(X2_train, y_train)

x_min, x_max = (
    X.iloc[:, 2].min() - 0.5,
    X.iloc[:, 2].max() + 0.5
)

y_min, y_max = (
    X.iloc[:, 3].min() - 0.5,
    X.iloc[:,
