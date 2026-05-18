# AAI-540 MLOps — Team 5: Financial Fraud Detection

**AI-Powered Real-Time Financial Fraud & Risk Intelligence Platform**

Team 5 project workspace for AAI-540 MLOps. We build an end-to-end machine learning system that detects credit-card fraud, scores transaction risk, explains model decisions, and ships a Streamlit demo for real-time inference.

---

## Overview

Financial fraud is highly imbalanced (~0.17% fraudulent transactions). This platform:

- Trains a **Random Forest** classifier on the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset with **SMOTE** balancing
- Delivers **real-time fraud scoring** and tiered risk levels (LOW → CRITICAL)
- Provides an **agentic fraud investigator** layer with actionable analyst recommendations
- Explains predictions with **SHAP** (global) and **LIME** (local)
- Exports a **Streamlit** app for interactive demos

Full architecture, data schema, metrics, and deployment notes are in the [design document](financial_fraud_detection/DESIGN.md).

---

## Repository structure

```
aai-540-mlops-team5/
├── README.md
├── LICENSE
└── financial_fraud_detection/
    ├── DESIGN.md                    # System design & architecture
    ├── AI Fraud Detection Platform.pdf
    └── jupyter_notebook/
        └── Team_5_AI_Powered_Real_Time_Financial_Fraud_&_Risk_Intelligence_Platform.ipynb
```

---

## Quick start

### 1. Clone and open the notebook

```bash
git clone <repository-url>
cd aai-540-mlops-team5
```

Open and run all cells in:

`financial_fraud_detection/jupyter_notebook/Team_5_AI_Powered_Real_Time_Financial_Fraud_&_Risk_Intelligence_Platform.ipynb`

### 2. Dataset

Download [`creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it where the notebook expects it (Colab: `/content/creditcard.csv`; local: update the path in the **Load Data** cell).

### 3. Dependencies

The notebook installs core packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn shap lime streamlit matplotlib seaborn joblib
```

### 4. Run the Streamlit demo (after notebook)

The notebook generates `fraud_detection_model.pkl` and `app.py`. From that working directory:

```bash
streamlit run app.py
```

Ensure `creditcard.csv` is in the same folder as the app.

---

## Pipeline summary

| Phase | Description |
|-------|-------------|
| Data & EDA | Load 284k transactions; explore severe class imbalance |
| Feature engineering | `Amount_Log`, `Hour` from `Time` and `Amount` |
| Training | SMOTE on 50k subset → Random Forest (100 trees, max depth 10) |
| Evaluation | Accuracy 99.86%, **Recall 80.61%**, ROC-AUC **95.32%** |
| Risk intelligence | Probability thresholds → risk tiers + investigator alerts |
| XAI | SHAP summary, LIME local explanations, feature importance |
| Deployment | `joblib` model + Streamlit UI |

See [DESIGN.md](financial_fraud_detection/DESIGN.md) for diagrams, API contracts, and production roadmap.

---

## Model performance (test set)

| Metric | Value |
|--------|-------|
| Accuracy | 99.86% |
| Precision (fraud) | 56.43% |
| Recall (fraud) | 80.61% |
| F1 (fraud) | 66.39% |
| ROC-AUC | 95.32% |

Recall is prioritized so fewer fraudulent transactions are missed—typical for fraud operations where false negatives are costly.

---

## Key features

1. **Fraud detection ML pipeline** — preprocessing, SMOTE, Random Forest  
2. **Real-time fraud scoring** — `predict_transaction()` with probability and risk level  
3. **Risk Intelligence Engine** — CRITICAL / HIGH / MODERATE / LOW tiers  
4. **Agentic AI Fraud Investigator** — human-readable escalation guidance  
5. **Explainability** — SHAP, LIME, feature importance  
6. **Streamlit deployment** — interactive transaction scoring  

---

## Documentation

- **[Design document](financial_fraud_detection/DESIGN.md)** — architecture, data model, ML design, XAI, deployment, limitations, future work  
- **[Jupyter notebook](financial_fraud_detection/jupyter_notebook/Team_5_AI_Powered_Real_Time_Financial_Fraud_&_Risk_Intelligence_Platform.ipynb)** — full implementation (25 sections)  
- **PDF** — `financial_fraud_detection/AI Fraud Detection Platform.pdf`  

---

## Team

AAI-540 MLOps — **Team 5**

---

## License

See [LICENSE](LICENSE).
