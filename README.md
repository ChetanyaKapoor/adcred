# AdCred Survey Dashboard

This repository contains a fully functional, multi‑tab Streamlit dashboard for analyzing influencer marketing survey data.

## Features
- **Data Visualization**: 10 descriptive charts with dynamic filters.
- **Classification**: KNN, Decision Tree, Random Forest, Gradient Boosting – metrics table, confusion matrix, ROC.
- **Clustering**: K‑means with elbow chart, persona table, downloadable clustered data.
- **Association Rules**: Apriori with adjustable support/confidence, top‑10 rules.
- **Regression**: Linear, Ridge, Lasso, Decision Tree – metrics comparison, feature importances.

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. (Optional) Edit secrets and advanced settings as needed.

Data file is located in `data/adcred_survey_synthetic.csv`. Replace with your own dataset if desired.