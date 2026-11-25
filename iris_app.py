"""
Iris Flower Predictor - Streamlit app (KNN) with tabbed UI

Run:  python -m streamlit run iris_app.py
Requires: streamlit, scikit-learn, pandas, matplotlib, seaborn
"""

import matplotlib.pyplot as plt
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


def set_preset(values):
    for key, val in zip(KEYS, values):
        st.session_state[key] = val


@st.cache_data
def split_data():
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_data
def k_search(max_k=25):
    X_tr, _, y_tr, _ = split_data()
    scores = []
    for k in range(1, max_k + 1):
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        scores.append(cross_val_score(m, X_tr, y_tr, cv=5).mean())
    return scores


@st.cache_data
def evaluate(k: int):
    X_tr, X_te, y_tr, y_te = split_data()
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return cv, m.score(X_te, y_te), confusion_matrix(y_te, pred), \
        classification_report(y_te, pred, target_names=class_names, output_dict=True)


@st.cache_data
def make_pairplot():
    g = sns.pairplot(df, hue="species", corner=True, height=2.0)
    return g.figure


# ------------------------------------------------------------------- Sidebar
st.sidebar.title("🌸 Controls")
st.sidebar.subheader("Quick examples")
cols = st.sidebar.columns(3)
for col, (label, vals) in zip(cols, PRESETS.items()):
    col.button(label.split()[0], help=label, on_click=set_preset, args=(vals,),
               use_container_width=False)

st.sidebar.subheader("Flower measurements (cm)")
for key, name in zip(KEYS, feature_names):
    lo, hi = float(X[name].min()), float(X[name].max())
    if key not in st.session_state:
        st.session_state[key] = round(float(X[name].mean()), 1)
    st.sidebar.slider(name, round(lo - 0.5, 1), round(hi + 0.5, 1), step=0.1, key=key)
inputs = {name: st.session_state[key] for key, name in zip(KEYS, feature_names)}

st.sidebar.subheader("Model")
best_k = k_search().index(max(k_search())) + 1
k = st.sidebar.slider("k (neighbours)", 1, 25, best_k,
                      help=f"Cross-validation suggests k = {best_k}")

# --------------------------------------------------------------------- Model
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
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

tab_predict, tab_data, tab_perf = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "📈 Model Performance"]
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
        st.markdown("**Class probabilities**")
        for name, p in zip(class_names, proba):
            st.progress(float(p), text=f"{name.capitalize()} — {p:.0%}")

        st.markdown("**Your measurements**")
        m_cols = st.columns(4)
        for c, (name, val) in zip(m_cols, inputs.items()):
            c.metric(name.replace(" (cm)", "").replace(" ", "\n"), f"{val} cm")

    with right:
        x_feat = st.selectbox("X axis", feature_names, index=2)
        y_feat = st.selectbox("Y axis", feature_names, index=3)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        for i, name in enumerate(class_names):
            mask = y == i
            ax.scatter(X.loc[mask, x_feat], X.loc[mask, y_feat],
                       label=name, alpha=0.7)
        ax.scatter(inputs[x_feat], inputs[y_feat], c="red", marker="*", s=320,
                   edgecolor="k", label="your flower", zorder=5)
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with st.expander(f"🔎 See the {k} nearest training flowers"):
        scaler = model.named_steps["standardscaler"]
        knn = model.named_steps["kneighborsclassifier"]
        dist, idx = knn.kneighbors(scaler.transform(sample))
        nb = X.iloc[idx[0]].copy()
        nb["species"] = [class_names[i] for i in y.iloc[idx[0]]]
        nb["distance"] = dist[0].round(3)
        st.dataframe(nb, width="stretch")

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
    cv_acc, test_acc, cm, report = evaluate(k)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current k", k)
    c2.metric("5-fold CV accuracy", f"{cv_acc:.1%}")
    c3.metric("Test accuracy (20% held out)", f"{test_acc:.1%}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Choosing k")
        scores = k_search()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(range(1, 26), scores, marker="o")
        ax.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
        ax.axvline(k, color="green", linestyle=":", label=f"selected k = {k}")
        ax.set_xlabel("k")
        ax.set_ylabel("CV accuracy")
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
    rep = pd.DataFrame(report).T.round(3)
    st.dataframe(rep, width="stretch")

with st.expander("ℹ️ How does KNN work?"):
    st.write(
        "KNN classifies a flower by finding the k closest flowers in the "
        "training data (after scaling the features) and taking a majority vote "
        "among their species. The probabilities are the share of the k "
        "neighbours belonging to each species."
    )
