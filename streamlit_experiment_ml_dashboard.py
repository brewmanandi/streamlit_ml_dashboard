from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt
import streamlit as st
from joblib import dump, load
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

categories = ['alt.atheism', 'comp.graphics', 'rec.autos', 'sci.electronics', 'talk.politics.guns',]
models = []

@st.cache_data
def load_and_vectorize_dataset():
    ## Load Dataset
    news_groups = datasets.fetch_20newsgroups(categories=categories)
    X_train, X_test, Y_train, Y_test = train_test_split(news_groups.data, news_groups.target, train_size=0.8, random_state=123)

    ## Vectorize Data
    vectorizer = CountVectorizer(max_features=50_000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, Y_train, Y_test

X_train_vec, X_test_vec, Y_train, Y_test = load_and_vectorize_dataset()

def train_models(n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease):
    # Adjust the parameters for each classifier according to their support
    models.append(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease))
    models.append(DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease))
    models.append(GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease))

    # Fit each model
    for model in models:
        model.fit(X_train_vec, Y_train)

## Dashboard
st.title("Classifier :green[Experiments] :tea: :coffee:")
st.markdown("Try different meta parameters. Select widget values and submit model for training. Various ML metrics will be displayed after training. Change any value to retrain.")


if 'submitted' not in st.session_state:
    st.session_state.submitted = True

# Callback function to set 'submitted' to True
def set_submitted():
    st.session_state.submitted = True


col1, col2 = st.columns(2, gap="medium")


with col1:
    n_estimators = st.slider("No of Estimators", min_value=100, max_value=1000, on_change=set_submitted)
    max_depth = st.slider("Max Depth", min_value=2, max_value=20, on_change=set_submitted)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, on_change=set_submitted)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, on_change=set_submitted)
with col2:
    max_leaf_nodes = st.slider("Max Leaf Nodes", min_value=10, max_value=1000, value=None, on_change=set_submitted)
    min_impurity_decrease = st.slider("Min Impurity Decrease", min_value=0.0, max_value=0.5, step=0.01, on_change=set_submitted)
    max_features = st.selectbox("Max Features", options=["sqrt", "log2", None], on_change=set_submitted)    


if st.session_state.submitted:
    with col2:
        with st.spinner('Training models, please wait...'):
            models = []
            train_models(n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease)
            st.success('Model training completed!')

            file_name = "models.dat"
            buffer = BytesIO()
            joblib.dump(models, buffer)
            buffer.seek(0)

            st.download_button(
                label="Download Models",
                data=buffer,
                file_name="models.pkl",
                mime="application/octet-stream"
            )
            st.toast(f"Models Successfully Downloaded!", icon="💾")            

    
    if len(models) > 0:
        columns = st.columns(len(models), gap="medium")
        for i, model in enumerate(models):
            with columns[i]:
                st.markdown(f"# {model.__class__.__name__}")
                Y_test_preds = model.predict(X_test_vec)
                Y_train_preds = model.predict(X_train_vec)
                Y_test_probs = model.predict_proba(X_test_vec)

                col21, col22 = st.columns(2, gap="medium")
                with col21:
                    st.metric("Test Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_test, Y_test_preds)))
                with col22:
                    st.metric("Train Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_train, Y_train_preds)))

                st.markdown("### Confusion Matrix")
                conf_mat_fig = plt.figure(figsize=(6,6))
                ax1 = conf_mat_fig.add_subplot(111)
                skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1)
                st.pyplot(conf_mat_fig, use_container_width=True)

                st.markdown("### Classification Report:")
                st.code("=="+classification_report(Y_test, Y_test_preds, target_names=categories))

                st.markdown("### ROC & Precision-Recall Curves")
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    roc_fig = plt.figure(figsize=(6,6))
                    ax1 = roc_fig.add_subplot(111)
                    skplt.metrics.plot_roc(Y_test, Y_test_probs, ax=ax1)
                    st.pyplot(roc_fig, use_container_width=True)

                with col2:
                    pr_fig = plt.figure(figsize=(6,6))
                    ax1 = pr_fig.add_subplot(111)
                    skplt.metrics.plot_precision_recall(Y_test, Y_test_probs, ax=ax1)
                    st.pyplot(pr_fig, use_container_width=True)

    st.session_state.submitted = False



hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)