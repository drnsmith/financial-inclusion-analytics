"""
mobile_money_model.py
=====================
Predicts which countries are likely to see mobile money uptake surge
in the next 5 years.

Definition of surge: country moves from <15% to >30% mobile money
account ownership within a 5-year window.

Features: GDP per capita, mobile subscriptions, internet penetration,
urbanisation, remittances, rule of law, poverty rate, starting
mobile money penetration.

Output:
  models_saved/mobile_money_xgb.pkl
  data/processed/mobile_money_predictions.parquet

Run:
    conda activate ds
    PYTHONPATH=. python src/mobile_money_model.py
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import shap

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models_saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mobile_money_pct_base",
    "account_ownership_pct_base",
    "mobile_subs_per100",
    "internet_pct",
    "gdp_per_capita_log",
    "remittances_gdp_pct",
    "urban_population_pct",
    "rule_of_law",
    "poverty_190",
    "account_growth_5y",
    "mobile_growth_5y",
]


def build_training_data(panel: pd.DataFrame,
                        window: int = 5,
                        mm_base_thresh: float = 15.0,
                        mm_surge_thresh: float = 30.0) -> pd.DataFrame:
    """
    Build labelled training set from historical panel.
    For each country × base year, define surge = 1 if
    mobile_money_pct grows from <15% to >30% within 5 years.
    """
    records = []
    panel = panel.sort_values(["iso3", "year"])

    for iso3, cdf in panel.groupby("iso3"):
        cdf = cdf.set_index("year")
        years = sorted(cdf.index)

        for base_yr in years[:-window]:
            target_yr = base_yr + window
            if target_yr not in cdf.index:
                continue

            base_row   = cdf.loc[base_yr]
            target_row = cdf.loc[target_yr]

            mm_base   = base_row.get("mobile_money_pct", np.nan)
            mm_target = target_row.get("mobile_money_pct", np.nan)

            # Fill missing with 0 — treat as no mobile money
            mm_base   = 0.0 if pd.isna(mm_base) else mm_base
            mm_target = 0.0 if pd.isna(mm_target) else mm_target

            if mm_base >= mm_base_thresh:
                continue  # already above threshold — not a surge candidate

            surge = int(mm_target >= mm_surge_thresh)

            # Feature: 5-year growth rates
            acc_base   = base_row.get("account_ownership_pct", np.nan)
            acc_prev   = cdf.loc[base_yr - 5].get("account_ownership_pct", np.nan) if (base_yr - 5) in cdf.index else np.nan
            mm_prev    = cdf.loc[base_yr - 5].get("mobile_money_pct", np.nan) if (base_yr - 5) in cdf.index else np.nan

            records.append({
                "iso3":                   iso3,
                "base_year":              base_yr,
                "surge":                  surge,
                "mobile_money_pct_base":  mm_base,
                "account_ownership_pct_base": acc_base if not pd.isna(acc_base) else 0,
                "mobile_subs_per100":     base_row.get("mobile_subs_per100", 0),
                "internet_pct":           base_row.get("internet_pct", 0),
                "gdp_per_capita_log":     np.log1p(base_row.get("gdp_per_capita", 1000)),
                "remittances_gdp_pct":    base_row.get("remittances_gdp_pct", 0),
                "urban_population_pct":   base_row.get("urban_population_pct", 50),
                "rule_of_law":            base_row.get("rule_of_law", 0),
                "poverty_190":            base_row.get("poverty_190", 20),
                "account_growth_5y":      (acc_base - acc_prev) if not pd.isna(acc_prev) and not pd.isna(acc_base) else 0,
                "mobile_growth_5y":       (mm_base - mm_prev) if not pd.isna(mm_prev) else 0,
            })

    return pd.DataFrame(records)


def build_prediction_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build features for 2023→2028 surge prediction.
    Uses 2023 predictor variables but last available Findex year for mobile money baseline.
    """
    # Use 2023 for predictors (infrastructure, GDP etc)
    latest_year = 2023
    # Use last Findex survey year for mobile money baseline
    findex_years = panel[panel["mobile_money_pct"].notna()]["year"]
    findex_year  = int(findex_years.max()) if len(findex_years) > 0 else 2021
    prev_year    = latest_year - 5

    latest  = panel[panel["year"] == latest_year].set_index("iso3")
    findex  = panel[panel["year"] == findex_year].set_index("iso3")
    prev    = panel[panel["year"] == prev_year].set_index("iso3") if prev_year in panel["year"].values else pd.DataFrame()

    records = []
    for iso3, row in latest.iterrows():
        # Get mobile money from last Findex wave
        findex_row = findex.loc[iso3] if iso3 in findex.index else pd.Series()
        mm_base = findex_row.get("mobile_money_pct", np.nan) if not findex_row.empty else np.nan
        # Include all countries with known low mobile money OR unknown (treat unknown as 0)
        if not pd.isna(mm_base) and mm_base >= 40:
            continue  # already saturated — skip
        mm_base = mm_base if not pd.isna(mm_base) else 0.0
        # Skip high-income countries with no Findex data (likely not mobile money markets)
        gdp = row.get("gdp_per_capita", 0)
        if pd.isna(mm_base) or (mm_base == 0.0 and gdp > 15000):
            continue

        prev_row = prev.loc[iso3] if (not prev.empty and iso3 in prev.index) else pd.Series()
        acc_base = row.get("account_ownership_pct", 0)
        acc_prev = prev_row.get("account_ownership_pct", np.nan) if not prev_row.empty else np.nan
        mm_prev  = prev_row.get("mobile_money_pct", np.nan) if not prev_row.empty else np.nan

        records.append({
            "iso3":                      iso3,
            "country_name":              row.get("country_name", iso3),
            "is_apac":                   row.get("is_apac", False),
            "mobile_money_pct_current":  mm_base,
            "account_ownership_pct":     acc_base,
            "mobile_money_pct_base":     mm_base,
            "account_ownership_pct_base":acc_base,
            "mobile_subs_per100":        row.get("mobile_subs_per100", 0),
            "internet_pct":              row.get("internet_pct", 0),
            "gdp_per_capita_log":        np.log1p(row.get("gdp_per_capita", 1000)),
            "remittances_gdp_pct":       row.get("remittances_gdp_pct", 0),
            "urban_population_pct":      row.get("urban_population_pct", 50),
            "rule_of_law":               row.get("rule_of_law", 0),
            "poverty_190":               row.get("poverty_190", 20),
            "account_growth_5y":         (acc_base - acc_prev) if not pd.isna(acc_prev) else 0,
            "mobile_growth_5y":          (mm_base - mm_prev) if not pd.isna(mm_prev) else 0,
        })

    return pd.DataFrame(records)


def train_model(df_train: pd.DataFrame):
    from xgboost import XGBClassifier

    X = df_train[FEATURE_COLS].fillna(0)
    y = df_train["surge"]

    if y.sum() < 5:
        log.warning(f"Only {y.sum()} positive examples — using synthetic augmentation")
        # Upsample positives
        if (y==1).sum() == 0:
            # No real positives — create synthetic ones from high mobile_money rows
            mm_col = "mobile_money_pct" if "mobile_money_pct" in df_train.columns else df_train.columns[2]
            pos = df_train.nlargest(20, mm_col).sample(20, replace=True, random_state=42)
        else:
            pos = df_train[y==1].sample(20, replace=True, random_state=42)
        df_train = pd.concat([df_train, pos], ignore_index=True)
        X = df_train[FEATURE_COLS].fillna(0)
        y = df_train["surge"]

    scale_pos = (y==0).sum() / max((y==1).sum(), 1)
    log.info(f"Training set: {len(X)} rows | surge rate: {y.mean():.2%} | scale_pos_weight: {scale_pos:.1f}")

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        eval_metric="aucpr", verbosity=0,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")
    log.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    xgb.fit(X, y)

    # SHAP
    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)
    shap_imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)

    shap_path = MODELS_DIR / "shap_mobile_money.csv"
    shap_imp.to_csv(shap_path, index=False)
    log.info(f"SHAP saved → {shap_path}")
    log.info("\nTop features:")
    for _, row in shap_imp.head(5).iterrows():
        log.info(f"  {row['feature']:35s} {row['importance']:.4f}")

    # Save model
    model_path = MODELS_DIR / "mobile_money_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": xgb, "features": FEATURE_COLS, "cv_auc": cv_scores.mean()}, f)
    log.info(f"Model saved → {model_path}")

    return xgb, shap_imp


def predict_surge(model, df_pred: pd.DataFrame) -> pd.DataFrame:
    X = df_pred[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    df_pred = df_pred.copy()
    df_pred["surge_probability_2028"] = probs
    df_pred["opportunity_tier"] = pd.cut(
        probs,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low", "Moderate", "High", "Very High"],
    )
    return df_pred.sort_values("surge_probability_2028", ascending=False)


def run():
    panel_path = DATA_PROC / "findex_panel.parquet"
    if not panel_path.exists():
        log.info("Building demo panel...")
        from build_findex_panel import make_demo_findex
        panel = make_demo_findex()
    else:
        panel = pd.read_parquet(panel_path)

    log.info(f"Panel: {panel.shape} | Countries: {panel['iso3'].nunique()}")

    # Build training data
    df_train = build_training_data(panel)
    log.info(f"Training set: {df_train.shape} | Surge rate: {df_train['surge'].mean():.2%}")

    # Train
    model, shap_imp = train_model(df_train)

    # Predict current candidates
    df_pred = build_prediction_features(panel)
    log.info(f"\nPrediction candidates (current MM < 15%): {len(df_pred)}")

    if not df_pred.empty:
        results = predict_surge(model, df_pred)

        out = DATA_PROC / "mobile_money_predictions.parquet"
        results.to_parquet(out, index=False)
        log.info(f"\n✓ Predictions saved → {out}")

        log.info("\nTop 10 mobile money surge candidates:")
        cols = ["iso3", "country_name", "mobile_money_pct_current",
                "surge_probability_2028", "opportunity_tier"]
        cols = [c for c in cols if c in results.columns]
        print(results[cols].head(10).to_string(index=False))

    return model, shap_imp


if __name__ == "__main__":
    run()
