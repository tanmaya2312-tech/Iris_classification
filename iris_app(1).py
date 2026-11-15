"""
Iris Flower Predictor - Streamlit app (KNN only)

Features:
  - Live prediction with confidence and a "near the boundary" warning
  - Distance metric selector (Euclidean / Manhattan / Minkowski) and
    uniform vs distance-weighted voting
  - Decision boundary plot that changes with k, metric and weighting
  - Batch prediction from an uploaded CSV
  - Data exploration and model performance tabs

Run:  python -m streamlit run iris_app.py
Requires: streamlit, scikit-learn, pandas, matplotlib, seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Iris KNN Predictor", page_icon="🌸", layout="wide")

# ------------------------------------------------------------------- Styling
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: .3rem 0 0 0; opacity: .9; }
    .result-card {
        padding: 1.4rem 1.6rem; border-radius: 16px; text-align: center;
        color: white; margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,.15);
    }
    .result-card .emoji { font-size: 3rem; }
    .result-card .name  { font-size: 1.9rem; font-weight: 700; }
    .result-card .conf  { font-size: 1rem; opacity: .95; }
    </style>
    """,
    unsafe_allow_html=True,
)

SPECIES_STYLE = {
    "setosa":     {"emoji": "🌼", "color": "linear-gradient(120deg,#f7971e,#ffd200)"},
    "versicolor": {"emoji": "💜", "color": "linear-gradient(120deg,#7f00ff,#e100ff)"},
    "virginica":  {"emoji": "🌺", "color": "linear-gradient(120deg,#11998e,#38ef7d)"},
}

# Distance metric options: label -> (metric, p)
METRICS = {
    "Euclidean": ("minkowski", 2),
    "Manhattan": ("manhattan", 1),
    "Minkowski (p=3)": ("minkowski", 3),
}
WEIGHTS = {"Uniform (equal vote)": "uniform", "Distance-weighted": "distance"}

# ---------------------------------------------------------------------- Data
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
class_names = list(iris.target_names)
feature_names = list(X.columns)
df = X.copy()
df["species"] = y.map(dict(enumerate(class_names)))

KEYS = ["sl", "sw", "pl", "pw"]
PRESETS = {
    "🌼 Setosa":     [5.0, 3.4, 1.5, 0.2],
    "💜 Versicolor": [5.9, 2.8, 4.3, 1.3],
    "🌺 Virginica":  [6.6, 3.0, 5.6, 2.0],
}
SHORT_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def set_preset(values):
    for key, val in zip(KEYS, values):
        st.session_state[key] = val


def make_model(k, metric_label, weights_label):
    metric, p = METRICS[metric_label]
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, metric=metric, p=p,
                             weights=WEIGHTS[weights_label]),
    )


@st.cache_data
def split_data():
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_data
def k_search(metric_label, weights_label, max_k=25):
    X_tr, _, y_tr, _ = split_data()
    return [
        cross_val_score(make_model(k, metric_label, weights_label),
                        X_tr, y_tr, cv=5).mean()
        for k in range(1, max_k + 1)
    ]


@st.cache_data
def evaluate(k, metric_label, weights_label):
    X_tr, X_te, y_tr, y_te = split_data()
    m = make_model(k, metric_label, weights_label)
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return (cv, m.score(X_te, y_te), confusion_matrix(y_te, pred),
            classification_report(y_te, pred, target_names=class_names,
                                  output_dict=True))


@st.cache_data
def compare_settings(k):
    """CV and test accuracy for every metric / weighting combination at this k."""
    rows = []
    for metric_label in METRICS:
        for weights_label in WEIGHTS:
            cv, test, _, _ = evaluate(k, metric_label, weights_label)
            rows.append({"Distance metric": metric_label,
                         "Voting": weights_label.split(" (")[0],
                         "CV accuracy": round(cv, 4),
                         "Test accuracy": round(test, 4)})
    return pd.DataFrame(rows)


@st.cache_data
def make_pairplot():
    return sns.pairplot(df, hue="species", corner=True, height=2.0).figure


def normalise_columns(frame):
    """Accept 'sepal_length', 'sepal length (cm)', 'Sepal Length' etc."""
    frame = frame.copy()
    frame.columns = [
        str(c).strip().lower().replace(" (cm)", "").replace(" ", "_")
        for c in frame.columns
    ]
    return frame


# ------------------------------------------------------------------- Sidebar
st.sidebar.title("🌸 Controls")
st.sidebar.subheader("Quick examples")
for col, (label, vals) in zip(st.sidebar.columns(3), PRESETS.items()):
    col.button(label.split()[0], help=label, on_click=set_preset, args=(vals,))

st.sidebar.subheader("Flower measurements (cm)")
for key, name in zip(KEYS, feature_names):
    lo, hi = float(X[name].min()), float(X[name].max())
    if key not in st.session_state:
        st.session_state[key] = round(float(X[name].mean()), 1)
    st.sidebar.slider(name, round(lo - 0.5, 1), round(hi + 0.5, 1), step=0.1, key=key)
inputs = {name: st.session_state[key] for key, name in zip(KEYS, feature_names)}

st.sidebar.subheader("KNN settings")
default_scores = k_search("Euclidean", "Uniform (equal vote)")
default_k = default_scores.index(max(default_scores)) + 1
k = st.sidebar.slider("k (neighbours)", 1, 25, default_k,
                      help=f"Cross-validation suggests k = {default_k} "
                           "for Euclidean distance with uniform voting")
metric_label = st.sidebar.selectbox("Distance metric", list(METRICS))
weights_label = st.sidebar.radio("Voting", list(WEIGHTS))

# --------------------------------------------------------------------- Model
model = make_model(k, metric_label, weights_label)
model.fit(X, y)
sample = pd.DataFrame([inputs])
pred = int(model.predict(sample)[0])
proba = model.predict_proba(sample)[0]

# -------------------------------------------------------------------- Header
st.markdown(
    """
    <div class="hero">
      <h1>🌸 Iris Flower Predictor</h1>
      <p>K-Nearest Neighbors classifier · 150 samples · 4 features · 3 species</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_data, tab_perf = st.tabs(
    ["🔮 Predict", "📁 Batch Prediction", "📊 Explore Data", "📈 Model Performance"]
)

# --------------------------------------------------------------- Tab: Predict
with tab_predict:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        species = class_names[pred]
        style = SPECIES_STYLE[species]
        st.markdown(
            f"""
            <div class="result-card" style="background:{style['color']}">
              <div class="emoji">{style['emoji']}</div>
              <div class="name">Iris {species.capitalize()}</div>
              <div class="conf">Confidence: {proba[pred]:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top2 = np.argsort(proba)[::-1][:2]
        if proba[top2[0]] < 0.8:
            a, b = class_names[top2[0]], class_names[top2[1]]
            st.warning(
                f"⚠️ This flower is near the **{a.capitalize()} / "
                f"{b.capitalize()}** boundary, so the prediction is less certain."
            )

        st.markdown("**Class probabilities**")
        for name, p in zip(class_names, proba):
            st.progress(float(p), text=f"{name.capitalize()} — {p:.0%}")

        with st.expander(f"🔎 See the {k} nearest training flowers"):
            scaler = model.named_steps["standardscaler"]
            knn = model.named_steps["kneighborsclassifier"]
            dist, idx = knn.kneighbors(scaler.transform(sample))
            nb = X.iloc[idx[0]].copy()
            nb["species"] = [class_names[i] for i in y.iloc[idx[0]]]
            nb["distance"] = dist[0].round(3)
            st.dataframe(nb, width="stretch")

    with right:
        st.markdown("**Decision boundary**")
        c1, c2 = st.columns(2)
        x_feat = c1.selectbox("X axis", feature_names, index=2)
        y_feat = c2.selectbox("Y axis", feature_names, index=3)
        if x_feat == y_feat:
            st.info("Choose two different features to draw the boundary.")
        else:
            m2d = make_model(k, metric_label, weights_label)
            m2d.fit(X[[x_feat, y_feat]].values, y)
            x_min, x_max = X[x_feat].min() - 0.5, X[x_feat].max() + 0.5
            y_min, y_max = X[y_feat].min() - 0.5, X[y_feat].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                                 np.linspace(y_min, y_max, 250))
            zz = m2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.contourf(xx, yy, zz, alpha=0.2, cmap="viridis",
                        levels=[-0.5, 0.5, 1.5, 2.5])
            for i, name in enumerate(class_names):
                mask = y == i
                ax.scatter(X.loc[mask, x_feat], X.loc[mask, y_feat],
                           label=name, alpha=0.8, edgecolor="k", linewidth=0.3)
            ax.scatter(inputs[x_feat], inputs[y_feat], c="red", marker="*",
                       s=340, edgecolor="k", label="your flower", zorder=5)
            ax.set_xlabel(x_feat)
            ax.set_ylabel(y_feat)
            ax.set_title(f"k = {k} · {metric_label} · "
                         f"{weights_label.split(' (')[0].lower()} voting")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            st.caption("Shaded regions show which species KNN would predict "
                       "at each point using only these two features. "
                       "Change k, the metric or the voting in the sidebar to "
                       "see the regions change.")

# ---------------------------------------------------------- Tab: Batch upload
with tab_batch:
    st.subheader("Predict many flowers from a CSV")
    st.write("Upload a CSV with these columns: "
             "`sepal_length`, `sepal_width`, `petal_length`, `petal_width`. "
             "An optional `species` column lets the app check its accuracy.")

    template = df.sample(5, random_state=1).drop(columns="species")
    template.columns = SHORT_NAMES
    st.download_button("⬇️ Download a sample CSV", template.to_csv(index=False),
                       file_name="sample_flowers.csv", mime="text/csv")

    upload = st.file_uploader("Upload your CSV", type="csv")
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            data = normalise_columns(raw)
            missing = [c for c in SHORT_NAMES if c not in data.columns]
            if missing:
                st.error(f"Missing column(s): {', '.join(missing)}")
            else:
                feats = data[SHORT_NAMES].apply(pd.to_numeric, errors="coerce")
                bad_rows = feats.isnull().any(axis=1)
                if bad_rows.any():
                    st.warning(f"Skipped {int(bad_rows.sum())} row(s) with "
                               "missing or non-numeric values.")
                feats = feats[~bad_rows]
                feats.columns = feature_names  # match the trained model
                preds = model.predict(feats)
                probs = model.predict_proba(feats)

                out = raw.loc[feats.index].copy()
                out["predicted_species"] = [class_names[p] for p in preds]
                out["confidence"] = probs.max(axis=1).round(3)
                st.success(f"Predicted {len(out)} flowers "
                           f"(k={k}, {metric_label}, "
                           f"{weights_label.split(' (')[0].lower()} voting).")
                st.dataframe(out, width="stretch")
                st.bar_chart(out["predicted_species"].value_counts())
                st.download_button("⬇️ Download predictions",
                                   out.to_csv(index=False),
                                   file_name="iris_predictions.csv",
                                   mime="text/csv")

                if "species" in data.columns:
                    truth = data.loc[feats.index, "species"].astype(str).str.lower()
                    acc = (truth.values == out["predicted_species"].values).mean()
                    st.metric("Accuracy against your 'species' column", f"{acc:.1%}")
                    st.caption("The model was trained on the full Iris dataset, "
                               "so this is optimistic if your file contains "
                               "the same flowers.")
        except Exception as exc:  # unreadable file, wrong format, etc.
            st.error(f"Could not read that file: {exc}")

# ------------------------------------------------------------ Tab: Explore
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df))
    c2.metric("Features", len(feature_names))
    c3.metric("Classes", len(class_names))

    st.subheader("Dataset")
    st.dataframe(df, width="stretch", height=260)
    st.download_button("⬇️ Download as CSV", df.to_csv(index=False),
                       file_name="iris.csv", mime="text/csv")

    st.subheader("Summary statistics by species")
    st.dataframe(df.groupby("species").mean().round(2), width="stretch")

    st.subheader("Feature relationships")
    st.pyplot(make_pairplot())

# ---------------------------------------------------------------- Tab: Perf
with tab_perf:
    cv_acc, test_acc, cm, report = evaluate(k, metric_label, weights_label)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current k", k)
    c2.metric("5-fold CV accuracy", f"{cv_acc:.1%}")
    c3.metric("Test accuracy (20% held out)", f"{test_acc:.1%}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Choosing k")
        scores = k_search(metric_label, weights_label)
        best_k = scores.index(max(scores)) + 1
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(range(1, 26), scores, marker="o")
        ax.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
        ax.axvline(k, color="green", linestyle=":", label=f"selected k = {k}")
        ax.set_xlabel("k")
        ax.set_ylabel("CV accuracy")
        ax.set_title(f"{metric_label}, {weights_label.split(' (')[0].lower()} voting")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with right:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)

    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    st.subheader(f"Effect of distance metric and voting (k = {k})")
    st.dataframe(compare_settings(k), width="stretch", hide_index=True)
    st.caption("All rows use KNN. Only the distance metric and the way "
               "neighbours vote are changed.")

with st.expander("ℹ️ How does KNN work?"):
    st.write(
        "KNN classifies a flower by finding the k closest flowers in the "
        "training data (after scaling the features) and taking a vote among "
        "their species. With uniform voting every neighbour counts equally; "
        "with distance-weighted voting closer neighbours count more. The "
        "probabilities are the (weighted) share of neighbours in each species."
    )
