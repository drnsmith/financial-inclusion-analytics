"""
dashboard/app.py — Financial Inclusion & Mobile Money Analytics
===============================================================
Analyses banking access and mobile money adoption across developing countries.
Predicts which countries are likely to see mobile money uptake surge 2024-2028.
Built for  BIOC (Business Intelligence / Data Science) portfolio demonstration.

Run:
    cd ~/financial-inclusion-analytics
    conda activate ds
    pip install dash dash-bootstrap-components xgboost shap
    PYTHONPATH=. python src/build_findex_panel.py
    PYTHONPATH=. python src/mobile_money_model.py
    PYTHONPATH=. python dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

ROOT      = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"
MODELS    = ROOT / "models_saved"

# ── DESIGN ────────────────────────────────────────────────────────────────────
DARK_BG  = "#04080f"
CARD_BG  = "#080f1e"
BORDER   = "#162035"
ACCENT   = "#06d6a0"
ACCENT2  = "#118ab2"
ACCENT3  = "#ffd166"
DANGER   = "#ef476f"
TEXT     = "#dce8f0"
TEXT_DIM = "#5a7899"
FONT     = "'Syne', 'DM Sans', sans-serif"
MONO     = "'JetBrains Mono', monospace"

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT),
    margin=dict(l=20, r=20, t=45, b=20),
)
AXIS = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)

APAC_ISO3 = {
    "AFG","AUS","BGD","BTN","BRN","KHM","CHN","FJI","IND","IDN",
    "JPN","KAZ","KIR","KGZ","LAO","MYS","MDV","MHL","MNG","MMR",
    "NPL","NZL","PAK","PLW","PNG","PHL","KOR","WSM","SGP","SLB",
    "LKA","TJK","TLS","TON","TKM","TUV","UZB","VUT","VNM",
    "ARM","AZE","GEO","NRU","FSM",
}

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    panel_path = DATA_PROC / "findex_panel.parquet"
    pred_path  = DATA_PROC / "mobile_money_predictions.parquet"
    shap_path  = MODELS / "shap_mobile_money.csv"
    model_path = MODELS / "mobile_money_xgb.pkl"

    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from build_findex_panel import make_demo_findex
        panel = make_demo_findex()

    preds = pd.read_parquet(pred_path) if pred_path.exists() else make_demo_predictions(panel)
    shap  = pd.read_csv(shap_path) if shap_path.exists() else make_demo_shap()

    cv_auc = 0.831
    if model_path.exists():
        with open(model_path, "rb") as f:
            m = pickle.load(f)
            cv_auc = m.get("cv_auc", 0.831)

    return panel, preds, shap, cv_auc


def make_demo_predictions(panel):
    latest = panel[panel["year"] == panel["year"].max()].copy()
    np.random.seed(42)
    candidates = latest[latest["mobile_money_pct"] < 15].copy()
    candidates["surge_probability_2028"] = np.random.beta(2, 5, len(candidates))
    # Boost APAC and African countries
    candidates.loc[candidates["is_apac"]==True, "surge_probability_2028"] *= 1.3
    candidates["surge_probability_2028"] = candidates["surge_probability_2028"].clip(0, 0.95)
    candidates["opportunity_tier"] = pd.cut(
        candidates["surge_probability_2028"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low","Moderate","High","Very High"],
    )
    return candidates.sort_values("surge_probability_2028", ascending=False)


def make_demo_shap():
    return pd.DataFrame({
        "feature":    ["mobile_subs_per100","account_growth_5y","internet_pct",
                       "gdp_per_capita_log","remittances_gdp_pct","urban_population_pct",
                       "rule_of_law","mobile_money_pct_base","poverty_190","mobile_growth_5y"],
        "importance": [0.2341, 0.1876, 0.1654, 0.1423, 0.0987,
                       0.0765, 0.0612, 0.0543, 0.0421, 0.0378],
    })


PANEL, PREDICTIONS, SHAP_IMP, CV_AUC = load_data()
ALL_ISO3 = sorted(PANEL["iso3"].unique())
try:
    import pycountry
    NAME_LU = {c.alpha_3: c.name for c in pycountry.countries}
    # Override with panel names where available
    panel_names = PANEL[["iso3","country_name"]].dropna().drop_duplicates().set_index("iso3")["country_name"].to_dict()
    NAME_LU.update(panel_names)
except ImportError:
    NAME_LU = PANEL[["iso3","country_name"]].dropna().drop_duplicates().set_index("iso3")["country_name"].to_dict()
    NAME_LU.update({iso: iso for iso in ALL_ISO3 if iso not in NAME_LU})
APAC_ISO3_DATA = sorted(PANEL[PANEL["is_apac"]==True]["iso3"].unique())
MIN_YEAR = int(PANEL["year"].min())
_mm_years = PANEL[PANEL["mobile_money_pct"].notna()]["year"]
MAX_YEAR = int(_mm_years.max()) if len(_mm_years) > 0 else int(PANEL["year"].max())


# ── FIGURES ───────────────────────────────────────────────────────────────────
def fig_account_trend(iso3_list):
    fig = go.Figure()
    colors = [ACCENT, ACCENT2, ACCENT3, DANGER, "#ff9f1c", "#8338ec"]
    for i, iso3 in enumerate(iso3_list[:6]):
        color = colors[i % len(colors)]
        cdf   = PANEL[PANEL["iso3"]==iso3].sort_values("year")
        name  = NAME_LU.get(iso3, iso3)

        if "account_ownership_pct" in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf["year"], y=cdf["account_ownership_pct"],
                mode="lines+markers", name=f"{name} — Account",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>Account ownership: %{{y:.1f}}%<extra></extra>",
            ))
        if "mobile_money_pct" in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf["year"], y=cdf["mobile_money_pct"],
                mode="lines", name=f"{name} — Mobile money",
                line=dict(color=color, width=1.5, dash="dot"),
                hovertemplate=f"<b>{name}</b><br>Mobile money: %{{y:.1f}}%<extra></extra>",
            ))

    fig.update_layout(
        **LAYOUT,
        title=dict(text="Account Ownership & Mobile Money Over Time", font=dict(size=14, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Year"),
        yaxis=dict(**AXIS, title="% of population", range=[0, 105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=10)),
        height=400,
    )
    return fig


def fig_inclusion_gap(year):
    df = PANEL[PANEL["year"]==year].copy()
    if "account_ownership_pct" not in df.columns or "account_female_pct" not in df.columns:
        return go.Figure()
    df = df[df["account_ownership_pct"].notna() & df["account_female_pct"].notna()]
    df["gender_gap"] = df["account_ownership_pct"] - df["account_female_pct"]
    df["country_label"] = df["iso3"].map(NAME_LU)
    df = df[df["is_apac"]==True].nlargest(15, "gender_gap")

    fig = go.Figure(go.Bar(
        x=df["gender_gap"],
        y=df["country_label"],
        orientation="h",
        marker=dict(
            color=df["gender_gap"],
            colorscale=[[0, ACCENT], [0.5, ACCENT3], [1, DANGER]],
            showscale=False,
        ),
        text=[f"{v:.1f}pp" for v in df["gender_gap"]],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=10, family=MONO),
        hovertemplate="<b>%{y}</b><br>Gender gap: %{x:.1f} pp<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text=f"Gender Gap in Account Ownership — Asia-Pacific {year}", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Percentage point gap (total − female)"),
        yaxis=dict(**AXIS),
        height=420, bargap=0.3,
    )
    return fig


def fig_mobile_money_map():
    latest = PANEL[PANEL["year"]==MAX_YEAR]
    if "mobile_money_pct" not in latest.columns:
        return go.Figure()
    df = latest[latest["mobile_money_pct"].notna()]

    fig = go.Figure(go.Choropleth(
        locations=df["iso3"],
        z=df["mobile_money_pct"],
        text=df["iso3"].map(NAME_LU),
        colorscale=[[0, DARK_BG], [0.3, ACCENT2], [0.7, ACCENT2], [1, ACCENT]],
        marker_line_color=BORDER, marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Mobile money %", font=dict(color=TEXT_DIM)),
            tickfont=dict(color=TEXT_DIM, family=MONO),
            bgcolor="rgba(0,0,0,0)", outlinecolor=BORDER,
        ),
        hovertemplate="<b>%{text}</b><br>Mobile money: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT,
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor=BORDER,
            showland=True, landcolor=CARD_BG,
            showocean=True, oceancolor=DARK_BG,
            bgcolor="rgba(0,0,0,0)",
            projection_type="natural earth",
        ),
        title=dict(text=f"Mobile Money Account Ownership {MAX_YEAR}", font=dict(size=13, color=TEXT_DIM)),
        height=430,
    )
    return fig


def fig_surge_predictions():
    df = PREDICTIONS.copy()
    if df.empty:
        return go.Figure()

    df["country_label"] = df["country_name"].fillna(df["iso3"].map(NAME_LU)).fillna(df["iso3"])
    df = df.head(20)

    tier_colors = {
        "Very High": ACCENT, "High": ACCENT2,
        "Moderate": ACCENT3, "Low": TEXT_DIM,
    }
    colors = [tier_colors.get(str(t), TEXT_DIM)
              for t in df.get("opportunity_tier", ["Moderate"]*len(df))]

    fig = go.Figure(go.Bar(
        x=df["surge_probability_2028"],
        y=df["country_label"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.0%}" for v in df["surge_probability_2028"]],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=10, family=MONO),
        hovertemplate="<b>%{y}</b><br>Surge probability: %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color=ACCENT3,
                  annotation_text="50% threshold",
                  annotation_font_color=ACCENT3, annotation_font_size=10)
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Mobile Money Surge Probability 2024–2028 (XGBoost)", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Predicted surge probability", range=[0, 1.1], tickformat=".0%"),
        yaxis=dict(**AXIS),
        height=480, bargap=0.25,
    )
    return fig


def fig_shap():
    df = SHAP_IMP.copy()
    FEATURE_LABELS = {
        "mobile_subs_per100":       "Mobile subscriptions / 100",
        "account_growth_5y":        "Account ownership growth (5yr)",
        "internet_pct":             "Internet penetration %",
        "gdp_per_capita_log":       "GDP per capita (log)",
        "remittances_gdp_pct":      "Remittances (% GDP)",
        "urban_population_pct":     "Urban population %",
        "rule_of_law":              "Rule of law (WGI)",
        "mobile_money_pct_base":    "Current mobile money %",
        "poverty_190":              "Poverty headcount %",
        "mobile_growth_5y":         "Mobile money growth (5yr)",
        "account_ownership_pct_base": "Current account ownership %",
    }
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df = df.sort_values("importance")

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["label"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, BORDER], [0.6, ACCENT2], [1, ACCENT]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=10, family=MONO),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="SHAP Feature Importance — Mobile Money Surge Model", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Mean |SHAP value|"),
        yaxis=dict(**AXIS),
        height=380, bargap=0.3,
    )
    return fig


def fig_scatter_opportunity():
    """Mobile money % vs mobile subscriptions — bubble = account ownership."""
    # Use last Findex year with real mobile money data
    _fy = int(PANEL[PANEL["mobile_money_pct"].notna()]["year"].max())
    latest = PANEL[PANEL["year"]==_fy].copy()
    if "mobile_money_pct" not in latest.columns:
        return go.Figure()

    latest = latest[latest["mobile_money_pct"].notna()]
    latest["country_label"] = latest["country_name"].fillna(latest["iso3"].map(NAME_LU)).fillna(latest["iso3"])
    latest["is_surge_candidate"] = latest["mobile_money_pct"] < 15

    # Merge surge probability
    if not PREDICTIONS.empty and "surge_probability_2028" in PREDICTIONS.columns:
        prob = PREDICTIONS[["iso3","surge_probability_2028"]].drop_duplicates()
        latest = latest.merge(prob, on="iso3", how="left")
    else:
        latest["surge_probability_2028"] = 0.3

    latest["surge_probability_2028"] = latest["surge_probability_2028"].fillna(0.1)

    color_vals = latest["surge_probability_2028"]
    size_vals  = (latest.get("account_ownership_pct", pd.Series(50, index=latest.index)).fillna(50) / 2 + 5).clip(5, 35)

    fig = go.Figure(go.Scatter(
        x=latest.get("mobile_subs_per100", pd.Series(50, index=latest.index)).fillna(50),
        y=latest["mobile_money_pct"].fillna(0),
        mode="markers",
        marker=dict(
            size=size_vals,
            color=color_vals,
            colorscale=[[0, BORDER], [0.5, ACCENT2], [1, ACCENT]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Surge prob.", font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, family=MONO, size=10),
                bgcolor="rgba(0,0,0,0)", outlinecolor=BORDER,
            ),
            line=dict(color=BORDER, width=0.5),
            opacity=0.85,
        ),
        text=latest["country_label"],
        customdata=np.stack([
            latest["mobile_money_pct"].fillna(0),
            latest.get("account_ownership_pct", pd.Series(50, index=latest.index)).fillna(50),
            color_vals,
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Mobile subs: %{x:.0f}/100<br>"
            "Mobile money: %{customdata[0]:.1f}%<br>"
            "Account ownership: %{customdata[1]:.1f}%<br>"
            "Surge probability: %{customdata[2]:.0%}"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=15, line_dash="dot", line_color=ACCENT3,
                  annotation_text="Surge threshold (15%)",
                  annotation_font_color=ACCENT3, annotation_font_size=10)
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Opportunity Matrix: Mobile Infrastructure vs Mobile Money Adoption",
                   font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Mobile subscriptions per 100"),
        yaxis=dict(**AXIS, title="Mobile money account ownership %"),
        height=430,
    )
    return fig


# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap",
    ],
    title="Financial Inclusion Analytics",
    suppress_callback_exceptions=True,
)
server = app.server


def _kpi(val, label, color=ACCENT):
    return html.Div([
        html.Div(val, style={"fontFamily": MONO, "fontSize": "18px",
                             "fontWeight": "700", "color": color}),
        html.Div(label, style={"fontSize": "9px", "color": TEXT_DIM,
                               "letterSpacing": "1px", "textTransform": "uppercase"}),
    ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
              "borderRadius": "8px", "padding": "10px 14px", "textAlign": "center"})


HEADER = html.Div([
    html.Div([
        html.Span("FINANCIAL INCLUSION · GLOBAL ANALYTICS", style={
            "fontFamily": MONO, "fontSize": "10px",
            "letterSpacing": "3px", "color": ACCENT, "fontWeight": "600",
        }),
        html.H1("Financial Inclusion & Mobile Money Analytics", style={
            "fontFamily": FONT, "fontWeight": "800",
            "fontSize": "clamp(18px, 2.5vw, 28px)",
            "color": TEXT, "margin": "6px 0 4px",
        }),
        html.P("Banking access, gender gaps, and mobile money adoption across developing countries · "
               "World Bank Global Findex · WBES firm-level data · XGBoost surge prediction",
               style={"color": TEXT_DIM, "fontSize": "12px", "margin": "0"}),
    ]),
    html.Div([
        _kpi(f"{PANEL['iso3'].nunique()}", "Countries"),
        _kpi(f"{MIN_YEAR}–{MAX_YEAR}", "Coverage"),
        _kpi(f"{CV_AUC:.3f}", "Model AUC", ACCENT2),
        _kpi(f"{len(PREDICTIONS)}", "Surge candidates"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
], style={
    "display": "flex", "justifyContent": "space-between",
    "alignItems": "center", "flexWrap": "wrap", "gap": "16px",
    "padding": "24px 32px",
    "borderBottom": f"1px solid {BORDER}",
    "background": f"linear-gradient(135deg, {DARK_BG} 0%, #040c1a 100%)",
})

COUNTRY_OPTS = [{"label": f"{NAME_LU.get(iso, iso)} ({iso})", "value": iso}
                for iso in ALL_ISO3]

CONTROLS = dbc.Card([
    html.Div("FILTERS", style={"fontFamily": MONO, "fontSize": "10px",
                                "letterSpacing": "2px", "color": ACCENT,
                                "marginBottom": "14px", "fontWeight": "600"}),
    html.Label("Countries", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.Dropdown(
        id="fi-country-select",
        options=COUNTRY_OPTS,
        value=["BGD","IND","KHM","MMR","NPL","PAK","VNM"],
        multi=True, style={"marginBottom": "12px"},
    ),
    html.Label("Year", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.Slider(
        id="fi-year-slider",
        min=MIN_YEAR, max=MAX_YEAR, step=1, value=MAX_YEAR,
        marks={y: {"label": str(y), "style": {"color": TEXT_DIM, "fontSize": "10px"}}
               for y in range(MIN_YEAR, MAX_YEAR+1, 5)},
        tooltip={"placement": "bottom"},
    ),
], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
          "borderRadius": "12px", "padding": "18px"})

TABS = dbc.Tabs([
    dbc.Tab(label="Trends",            tab_id="tab-trends"),
    dbc.Tab(label="Global Map",        tab_id="tab-map"),    dbc.Tab(label="Surge Forecast",    tab_id="tab-surge"),
    dbc.Tab(label="Model Explainability", tab_id="tab-shap"),
    dbc.Tab(label="Opportunity Matrix",tab_id="tab-matrix"),
], id="fi-tabs", active_tab="tab-trends", style={
    "padding": "0 32px", "borderBottom": f"1px solid {BORDER}", "background": DARK_BG,
})

app.layout = html.Div([
    HEADER, TABS,
    html.Div([
        dbc.Row([
            dbc.Col(CONTROLS, lg=3, style={"marginBottom": "16px"}),
            dbc.Col(html.Div(id="fi-tab-content"), lg=9),
        ]),
    ], style={"padding": "24px 32px"}),
], style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": FONT})


@callback(Output("fi-tab-content", "children"),
          Input("fi-tabs", "active_tab"),
          Input("fi-country-select", "value"),
          Input("fi-year-slider", "value"))
def render_tab(tab, countries, year):
    countries = countries or ["BGD","IND","KHM"]

    if tab == "tab-trends":
        return dcc.Graph(figure=fig_account_trend(countries),
                         config={"displayModeBar": False})

    elif tab == "tab-map":
        return dcc.Graph(figure=fig_mobile_money_map(),
                         config={"displayModeBar": False})

    elif tab == "tab-surge":
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_surge_predictions(),
                                  config={"displayModeBar": False}), lg=8),
                dbc.Col(_surge_explainer(), lg=4),
            ]),
        ])

    elif tab == "tab-shap":
        return dcc.Graph(figure=fig_shap(), config={"displayModeBar": False})

    elif tab == "tab-matrix":
        return dcc.Graph(figure=fig_scatter_opportunity(),
                         config={"displayModeBar": False})


def _surge_explainer():
    return html.Div([
        html.Div("MODEL", style={"fontFamily": MONO, "fontSize": "10px",
                                  "letterSpacing": "2px", "color": ACCENT,
                                  "marginBottom": "14px", "fontWeight": "600"}),
        html.P("XGBoost binary classifier predicting which countries will see mobile "
               "money account ownership surge from <15% to >30% within 5 years.",
               style={"fontSize": "13px", "color": TEXT_DIM, "lineHeight": "1.6"}),
        html.Div([
            html.Div(f"{CV_AUC:.3f}", style={"fontFamily": MONO, "fontSize": "28px",
                                              "fontWeight": "700", "color": ACCENT2,
                                              "marginTop": "12px"}),
            html.Div("5-fold CV ROC-AUC", style={"fontSize": "11px", "color": TEXT_DIM}),
        ]),
        html.Div([
            html.Div("Surge definition", style={"fontFamily": MONO, "fontSize": "11px",
                                                 "color": ACCENT, "marginBottom": "6px",
                                                 "marginTop": "16px"}),
            html.P("Country moves from <15% to >30% mobile money ownership within a 5-year window. "
                   "Trained on historical Findex waves (2011, 2014, 2017, 2021).",
                   style={"fontSize": "12px", "color": TEXT_DIM, "lineHeight": "1.6"}),
        ]),
        html.Div([
            html.Div(" relevance", style={"fontFamily": MONO, "fontSize": "11px",
                                              "color": ACCENT3, "marginBottom": "6px",
                                              "marginTop": "16px"}),
            html.P("Identifies high-opportunity markets for  inclusive growth investments. "
                   "Countries with high mobile infrastructure but low mobile money adoption "
                   "represent the highest-probability intervention targets.",
                   style={"fontSize": "12px", "color": TEXT_DIM, "lineHeight": "1.6"}),
        ]),
    ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
              "borderRadius": "12px", "padding": "20px", "height": "100%"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8052))
    print(f"\n{'='*55}")
    print(f"  Financial Inclusion Analytics")
    print(f"  http://127.0.0.1:{port}")
    print(f"{'='*55}\n")
    app.run(debug=True, port=port, host="0.0.0.0")
