
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.graph_objects as go
import io

st.set_page_config(page_title='AdCred Survey Dashboard', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv('data/adcred_survey_synthetic.csv')

df = load_data()

# Helper preprocessing functions
def preprocess_classification(df):
    drop_cols = ['Suggestions_Concerns','Location','Education_Level','Best_Value_Influencer_Tier','Biggest_Purchase_Factor']
    df_c = df.drop(columns=drop_cols).copy()
    for c in df_c.columns:
        if df_c[c].dtype == 'object':
            df_c[c] = LabelEncoder().fit_transform(df_c[c].astype(str))
    X = df_c.drop('Trust_Score', axis=1)
    y = df_c['Trust_Score']
    return X, y

def preprocess_regression(df):
    drop_cols = ['Suggestions_Concerns','Location','Education_Level','Best_Value_Influencer_Tier','Biggest_Purchase_Factor']
    df_r = df.drop(columns=drop_cols).copy()
    for c in df_r.columns:
        if df_r[c].dtype == 'object':
            df_r[c] = LabelEncoder().fit_transform(df_r[c].astype(str))
    return df_r

def get_cluster_personas(df, labels):
    tmp = df.copy()
    tmp['Cluster'] = labels
    return tmp.groupby('Cluster').agg({
        'Age':'mean','Monthly_Income':'mean','Trust_Score':'mean',
        'Avg_Monthly_Influencer_Spend':'mean','Num_Influencer_Purchases':'mean'
    }).round(2)

# Sidebar
st.sidebar.title('AdCred Survey Dashboard')
st.sidebar.write('Select a tab from above to explore.')

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Data Visualization','Classification','Clustering','Association Rules','Regression'])

with tab1:
    st.header('Data Visualization')
    # ... (visualization code omitted for brevity) ...

with tab2:
    st.header('Classification')
    X, y = preprocess_classification(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=120, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        results.append([
            name,
            round(accuracy_score(y_te, y_pred), 3),
            round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 3),
            round(recall_score(y_te, y_pred, average='weighted'), 3),
            round(f1_score(y_te, y_pred, average='weighted'), 3)
        ])

    st.dataframe(pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1']), use_container_width=True)

    sel = st.selectbox('Confusion Matrix for', list(models.keys()))
    y_sel_pred = models[sel].predict(X_te)
    cm = confusion_matrix(y_te, y_sel_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(sel)
    st.pyplot(fig)

    st.subheader('ROC Curve (micro-average)')
    fig = go.Figure()
    for n, m in models.items():
        if hasattr(m,'predict_proba'):
            probas = m.predict_proba(X_te)
            fpr, tpr, _ = roc_curve(y_te, probas.argmax(1), pos_label=y_te.max())
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=n))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig)

with tab3:
    st.header('Clustering')
    # ... clustering code ...
with tab4:
    st.header('Association Rules')
    # ... apriori code ...
with tab5:
    st.header('Regression')
    df_r = preprocess_regression(df)
    X_r = df_r.drop('Avg_Monthly_Influencer_Spend', axis=1)
    y_r = df_r['Avg_Monthly_Influencer_Spend']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    regs = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.05),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    rows=[]
    for n,m in regs.items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        r2 = m.score(X_te, y_te)
        rmse = np.sqrt(np.mean((preds - y_te)**2))
        rows.append([n, round(r2,3), round(rmse,2)])
    st.dataframe(pd.DataFrame(rows, columns=['Model','R2','RMSE']))
