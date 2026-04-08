# Financial Inclusion & Mobile Money Analytics

**Live demo:** https://drnsmith-financial-inclusion-analytics.hf.space

---

## Why I built this

Mobile money is one of the more remarkable development stories of the last fifteen years. M-Pesa launched in Kenya in 2007 with a simple premise — send money by SMS — and within a decade had moved a significant share of the Kenyan population into formal financial participation. The pattern has repeated across Sub-Saharan Africa and parts of South Asia, but not uniformly and not predictably.

I wanted to understand what makes the difference. Specifically: given what we know about a country's mobile infrastructure, economic conditions, and institutional environment today, can we predict which countries are most likely to see mobile money adoption surge from under 15% to over 30% within the next five years?

This is a tractable machine learning problem — but it's also a question with genuine policy relevance. Development finance institutions allocate capital toward financial inclusion interventions. If you can identify the countries where the enabling conditions exist but adoption hasn't yet taken off, you can direct that capital more precisely.

I also had a methodological interest here. The World Bank Findex survey only runs every three years, which means the standard approach of training and scoring on the same cross-section is invalid. I had to think carefully about how to construct a training set from historical surge events (2011→2016, 2014→2019, 2017→2022) and then apply the learned model to 2023 infrastructure data to score 2024–2028 surge probability. That temporal separation — training on past transitions, predicting future ones — is the right structure for this problem, and it took some care to implement correctly.

---

## What I built

An XGBoost binary classifier trained on 4,703 country-year observations drawn from four Findex survey waves. The target variable is a binary surge indicator: did this country move from under 15% to over 30% mobile money ownership within five years?

The model achieves 5-fold cross-validated ROC-AUC of 0.858 ± 0.017. The top predictors are mobile subscriptions per 100 population, poverty headcount, GDP per capita (log), internet penetration, and lagged mobile money growth. This ordering makes intuitive sense: mobile infrastructure is the necessary condition; poverty and GDP capture the demand side; internet penetration proxies for digital readiness.

I then scored all 268 countries in the panel using 2023 values of these predictor variables, generating surge probabilities for 2024–2028. The dashboard surfaces 178 current candidates — countries with mobile money below 15% or with evidence of recent emergence — ranked by surge probability.

The dashboard has five tabs: Trends, Global Map, Surge Forecast, Model Explainability, and Opportunity Matrix. The Opportunity Matrix is the tab I find most useful: it plots mobile subscriptions (x-axis) against mobile money ownership (y-axis), with bubble size encoding account ownership and colour encoding surge probability. Countries in the lower-right quadrant — high infrastructure, low adoption — are the highest-value intervention targets.

---

## Model

| Metric | Value |
|---|---|
| Algorithm | XGBoost binary classifier |
| Training observations | 4,703 country-years |
| Surge definition | <15% → >30% mobile money ownership within 5 years |
| Cross-validated ROC-AUC | 0.858 ± 0.017 |
| Surge candidates (2024–2028) | 178 countries |
| Top predictor | Mobile subscriptions per 100 |

---

## Data sources

| Source | Coverage | Variables |
|---|---|---|
| World Bank Global Findex | 268 countries, 2011–2022 | Account ownership, mobile money, digital payments |
| World Bank WDI | 268 countries, 2000–2023 | GDP, poverty, internet, mobile subscriptions |
| WBES Enterprise Surveys | 270,193 firms | Firm-level financial access barriers (k7, j12, b7) |

---

## Tech stack

- Python 3.11, XGBoost 2.0.3, SHAP 0.44.1, scikit-learn 1.4.0
- Plotly Dash 4.1.0, dash-bootstrap-components 2.0.4
- statsmodels, pandas, numpy, pyarrow
- Deployed on Render

---

## Run locally

```bash
git clone https://github.com/drnsmith/financial-inclusion-analytics.git
cd financial-inclusion-analytics
pip install -r requirements.txt

PYTHONPATH=. python src/build_findex_panel.py
PYTHONPATH=. python src/mobile_money_model.py

PYTHONPATH=. python dashboard/app.py
# → http://127.0.0.1:8052
```

---

## Project structure

```
financial-inclusion-analytics/
├── src/
│   ├── build_findex_panel.py   # ETL: WDI + Findex + WBES → parquet
│   └── mobile_money_model.py   # XGBoost training, SHAP, predictions
├── dashboard/
│   └── app.py                  # Plotly Dash, 5 tabs
├── models_saved/
│   ├── mobile_money_xgb.pkl
│   └── shap_mobile_money.csv
└── requirements.txt
```

---

## Author

Dr Natalya Smith — [github.com/drnsmith](https://github.com/drnsmith) · [medium.com/@NeverOblivious](https://medium.com/@NeverOblivious)
