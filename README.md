# Iris Flower Classification using KNN

Classifies iris flowers (Setosa, Versicolor, Virginica) from sepal and petal
measurements using the K-Nearest Neighbors algorithm, with an interactive
Streamlit web app.

## Files
- `iris_knn.py`  - full ML pipeline: EDA, scaling, choosing k with cross-validation, evaluation, plots
- `iris_app.py`  - Streamlit app (predict, explore data, model performance)
- `iris.csv`     - Iris dataset (150 samples, 4 features, 3 species)
- `requirements.txt` - Python dependencies

## Setup
```
pip install -r requirements.txt
```

## Run
```
python iris_knn.py
python -m streamlit run iris_app.py
```

## Method
1. Explore data and check class separability
2. Stratified 80/20 train/test split
3. Standardise features (KNN is distance-based)
4. Select k with 5-fold cross-validation
5. Evaluate with accuracy, confusion matrix, precision/recall/F1
6. Deploy as a Streamlit app
