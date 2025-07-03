
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import io

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("TasteMate_Synthetic_Consumer_Data.csv")

df = load_data()
st.set_page_config(layout="wide")
st.title("🍽️ TasteMate Kitchen – End-to-End Analytics Dashboard")

tabs = st.tabs(["📊 Data Visualization", "🤖 Classification", "🔍 Clustering", "🔗 Association Rules", "📈 Regression"])

# ----------------------- TAB 1: DATA VISUALIZATION --------------------------
with tabs[0]:
    st.header("📊 Descriptive Data Insights")

    st.subheader("Age Distribution")
    fig1 = px.histogram(df, x='Age', color='Gender')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Spend per Order by City")
    fig2 = px.box(df, x='City', y='Spend per Order', points="all")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cuisine Preferences")
    cuisine_df = df['Preferred Cuisines'].str.get_dummies(sep=', ')
    st.bar_chart(cuisine_df.sum().sort_values(ascending=False))

    st.subheader("Order Frequency vs Satisfaction")
    fig3 = px.box(df, x='Order Frequency', y='Satisfaction (1-5)', color='Order Frequency')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    fig4 = px.imshow(df[numeric_cols].corr(), text_auto=True)
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------- TAB 2: CLASSIFICATION --------------------------
with tabs[1]:
    st.header("🤖 Customer Retention Classification")

    df_class = df.dropna()
    le = LabelEncoder()
    df_class['Repeat Order'] = le.fit_transform(df_class['Repeat Order'])

    X = pd.get_dummies(df_class.drop(['Repeat Order'], axis=1), drop_first=True)
    y = df_class['Repeat Order']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            "Model": name,
            "Accuracy": report["accuracy"],
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-Score": report["weighted avg"]["f1-score"]
        })

    st.subheader("Model Comparison Table")
    st.dataframe(pd.DataFrame(results))

    model_choice = st.selectbox("Select model to view Confusion Matrix", list(models.keys()))
    selected_model = models[model_choice]
    selected_model.fit(X_train, y_train)
    y_pred_cm = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_cm)

    st.subheader(f"Confusion Matrix: {model_choice}")
    st.write(cm)

# ----------------------- TAB 3: CLUSTERING --------------------------
with tabs[2]:
    st.header("🔍 Customer Segmentation via Clustering")

    df_cluster = df.copy()
    cluster_features = pd.get_dummies(df_cluster[['Age', 'City', 'Order Frequency']], drop_first=True)
    k_slider = st.slider("Select Number of Clusters", 2, 10, 3)

    kmeans = KMeans(n_clusters=k_slider, random_state=42)
    clusters = kmeans.fit_predict(cluster_features)

    df_cluster['Cluster'] = clusters
    st.subheader("Cluster Centroids Overview")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features.columns))

    st.subheader("Download Cluster-Labeled Data")
    csv = df_cluster.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "clustered_data.csv", "text/csv")

# ----------------------- TAB 4: ASSOCIATION RULES --------------------------
with tabs[3]:
    st.header("🔗 Association Rule Mining")

    ar_cols = st.multiselect("Select Columns for Association Rules", ['Preferred Cuisines', 'Order Influence', 'Delivery Issues'])

    if len(ar_cols) >= 2:
        transactions = df[ar_cols].apply(lambda x: ','.join(x.dropna()), axis=1).str.split(',')
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)

        freq_items = apriori(df_trans, min_support=0.1, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

        st.subheader("Top 10 Rules by Confidence")
        st.dataframe(rules.sort_values(by='confidence', ascending=False).head(10))
    else:
        st.info("Please select at least 2 columns for association rules.")

# ----------------------- TAB 5: REGRESSION --------------------------
with tabs[4]:
    st.header("📈 Spend Prediction via Regression Models")

    df_reg = df.dropna()
    df_reg['Spend Num'] = df_reg['Spend per Order'].replace({'<₹100': 50, '₹100–200': 150, '₹200–400': 300, '₹400–600': 500, '₹600+': 800})

    y = df_reg['Spend Num']
    X = pd.get_dummies(df_reg.drop(['Spend Num', 'Spend per Order'], axis=1), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor()
    }

    insights = []
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        insights.append({'Model': name, 'R2 Score': round(r2, 3)})

    st.subheader("Regression Insights")
    st.dataframe(pd.DataFrame(insights))
