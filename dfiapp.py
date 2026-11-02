import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate

# OR-Tools is used for the inventory order recommendation.
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    cp_model = None
    ORTOOLS_AVAILABLE = False


# ------------------------- CONFIGURATION -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgb_dfio_model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")
HIST_PATH = os.path.join(DATA_DIR, "historical_backtest.csv")
FUTURE_PATH = os.path.join(DATA_DIR, "future_forecasts.csv")
METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.csv")

# Change these only if your exported files use different names.
DATE_COL = "date"
ACTUAL_COL = "actual"
PRED_COL = "predicted"
FORECAST_COL = "forecast"


# ------------------------- LOAD ARTIFACT / DATA -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. Export your trained LightGBM model with Joblib first."
    )

artifact = joblib.load(MODEL_PATH)
if isinstance(artifact, dict):
    MODEL = artifact.get("model", artifact.get("booster"))
    FEATURE_COLS = artifact.get("feature_cols", artifact.get("features", []))
    MODEL_TYPE = artifact.get("model_type", "LightGBM")
    TARGET = artifact.get("target", "sales")
else:
    MODEL = artifact
    FEATURE_COLS = []
    MODEL_TYPE = "LightGBM"
    TARGET = "sales"

if not os.path.exists(HIST_PATH):
    raise FileNotFoundError(f"Missing {HIST_PATH}")
if not os.path.exists(FUTURE_PATH):
    raise FileNotFoundError(f"Missing {FUTURE_PATH}")

hist = pd.read_csv(HIST_PATH, parse_dates=[DATE_COL])
future = pd.read_csv(FUTURE_PATH, parse_dates=[DATE_COL])

for frame, required, label in [
    (hist, [DATE_COL, "item_id", "store_id", ACTUAL_COL, PRED_COL], "historical_backtest.csv"),
    (future, [DATE_COL, "item_id", "store_id", FORECAST_COL], "future_forecasts.csv"),
]:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

for frame in (hist, future):
    for col in ["state_id", "cat_id", "dept_id"]:
        if col not in frame.columns:
            frame[col] = "All"
        frame[col] = frame[col].fillna("Unknown").astype(str)

for col in ["lower", "upper"]:
    if col not in hist.columns:
        hist[col] = np.nan
    if col not in future.columns:
        future[col] = np.nan

if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
else:
    metrics_df = pd.DataFrame(columns=["metric", "value"])


# ------------------------- METRICS -------------------------
def wape_score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - predicted).sum() / denom * 100) if denom else np.nan


def rmsse_score(actual, predicted, training_values=None, seasonality=1):
    """
    RMSSE = sqrt(mean squared forecast error / mean squared in-sample
    seasonal-naive difference). Supply training_values for a valid RMSSE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    numerator = np.mean((actual - predicted) ** 2)
    if training_values is None:
        return np.nan
    train = np.asarray(training_values, dtype=float)
    if len(train) <= seasonality:
        return np.nan
    denominator = np.mean((train[seasonality:] - train[:-seasonality]) ** 2)
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else np.nan


def metric_from_file(metric_name):
    if metrics_df.empty:
        return np.nan
    if {"metric", "value"}.issubset(metrics_df.columns):
        match = metrics_df[
            metrics_df["metric"].astype(str).str.upper() == metric_name.upper()
        ]
        if not match.empty:
            return pd.to_numeric(match.iloc[0]["value"], errors="coerce")
    if metric_name in metrics_df.columns:
        return pd.to_numeric(metrics_df.iloc[0][metric_name], errors="coerce")
    return np.nan


GLOBAL_WAPE = metric_from_file("WAPE")
if pd.isna(GLOBAL_WAPE):
    GLOBAL_WAPE = wape_score(hist[ACTUAL_COL], hist[PRED_COL])

GLOBAL_RMSSE = metric_from_file("RMSSE")
# RMSSE is intentionally not guessed from the validation set alone.
# Export a correctly calculated RMSSE in model_metrics.csv using training history.


# ------------------------- INVENTORY OPTIMIZATION -------------------------
def recommend_order_quantity(demand, on_hand, holding_cost, stockout_cost):
    """
    Choose integer order quantity q to minimize:
      holding_cost * leftover + stockout_cost * shortage
    using OR-Tools CP-SAT. Demand is the selected forecast demand.
    """
    demand = max(0, int(np.ceil(float(demand))))
    on_hand = max(0, int(np.ceil(float(on_hand))))
    holding_cost = max(0.01, float(holding_cost))
    stockout_cost = max(0.01, float(stockout_cost))

    # If OR-Tools is unavailable, use a transparent deterministic fallback.
    if not ORTOOLS_AVAILABLE:
        return max(0, demand - on_hand), "Fallback (OR-Tools not installed)"

    model = cp_model.CpModel()
    q = model.NewIntVar(0, max(1, demand + on_hand + 1000), "order_qty")
    leftover = model.NewIntVar(0, max(1, demand + on_hand + 1000), "leftover")
    shortage = model.NewIntVar(0, max(1, demand + on_hand + 1000), "shortage")

    # on_hand + q - demand = leftover - shortage
    model.Add(on_hand + q - demand == leftover - shortage)

    scale = 100  # CP-SAT objective coefficients must be integers.
    hc = int(round(holding_cost * scale))
    sc = int(round(stockout_cost * scale))
    model.Minimize(hc * leftover + sc * shortage)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return int(solver.Value(q)), "OR-Tools CP-SAT"
    return max(0, demand - on_hand), "Fallback (solver did not return a solution)"


# ------------------------- DASH APP -------------------------
app = Dash(__name__, title="Walmart M5 Forecasting")
server = app.server

NAVY = "#172554"
BLUE = "#2563EB"
LIGHT = "#F4F7FB"

def options_for(column, frame):
    vals = sorted(frame[column].dropna().astype(str).unique().tolist())
    return [{"label": "All", "value": "All"}] + [
        {"label": v, "value": v} for v in vals
    ]


def filter_frame(frame, state, store, item, category):
    out = frame.copy()
    if state != "All":
        out = out[out["state_id"].astype(str) == state]
    if store != "All":
        out = out[out["store_id"].astype(str) == store]
    if item != "All":
        out = out[out["item_id"].astype(str) == item]
    if category != "All":
        out = out[out["cat_id"].astype(str) == category]
    return out


def kpi_card(label, value, subtitle=""):
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "color": "#64748B"}),
            html.Div(value, style={"fontSize": "25px", "fontWeight": "700", "color": NAVY}),
            html.Div(subtitle, style={"fontSize": "11px", "color": "#64748B"}),
        ],
        style={
            "background": "white", "padding": "17px", "borderRadius": "12px",
            "boxShadow": "0 1px 5px rgba(0,0,0,.07)", "flex": "1",
            "minWidth": "160px"
        },
    )


app.layout = html.Div(
    style={"background": LIGHT, "minHeight": "100vh", "fontFamily": "Arial, sans-serif", "padding": "22px"},
    children=[
        html.Div(
            [
                html.Div("WALMART M5", style={"fontSize": "12px", "letterSpacing": "2px", "color": "#BFDBFE"}),
                html.H1("Sales Forecasting & Inventory Intelligence",
                        style={"margin": "5px 0", "fontSize": "27px"}),
                html.Div(
                    f"{MODEL_TYPE} model  |  Target: {TARGET}  |  Interactive business dashboard",
                    style={"color": "#DBEAFE", "fontSize": "13px"}
                ),
            ],
            style={"background": NAVY, "color": "white", "padding": "24px", "borderRadius": "14px", "marginBottom": "18px"},
        ),

        html.Div(
            [
                html.Div([html.Label("State"), dcc.Dropdown(id="state-filter", options=options_for("state_id", hist), value="All", clearable=False)]),
                html.Div([html.Label("Store"), dcc.Dropdown(id="store-filter", options=options_for("store_id", hist), value="All", clearable=False)]),
                html.Div([html.Label("Category"), dcc.Dropdown(id="category-filter", options=options_for("cat_id", hist), value="All", clearable=False)]),
                html.Div([html.Label("Item"), dcc.Dropdown(id="item-filter", options=options_for("item_id", hist), value="All", clearable=False)]),
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(150px, 1fr))", "gap": "12px",
                   "background": "white", "padding": "16px", "borderRadius": "12px", "marginBottom": "16px"},
        ),

        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),

        html.Div(
            [
                html.Div([html.H3("Historical: Actual vs Predicted"), dcc.Graph(id="historical-chart")],
                         style={"background": "white", "padding": "14px", "borderRadius": "12px"}),
                html.Div([html.H3("Future Forecast (next available horizon)"), dcc.Graph(id="future-chart")],
                         style={"background": "white", "padding": "14px", "borderRadius": "12px"}),
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))", "gap": "14px", "marginBottom": "16px"},
        ),

        html.Div(
            [
                html.Div([html.H3("Forecast by Store"), dcc.Graph(id="store-chart")],
                         style={"background": "white", "padding": "14px", "borderRadius": "12px"}),
                html.Div([html.H3("Forecast by Category"), dcc.Graph(id="category-chart")],
                         style={"background": "white", "padding": "14px", "borderRadius": "12px"}),
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))", "gap": "14px", "marginBottom": "16px"},
        ),

        html.Div(
            [
                html.H3("Inventory Order Recommendation"),
                html.Div("Select an item/store above, enter available inventory and cost assumptions.",
                         style={"color": "#64748B", "fontSize": "13px", "marginBottom": "12px"}),
                html.Div(
                    [
                        html.Div([html.Label("Current on-hand inventory"),
                                  dcc.Input(id="on-hand", type="number", min=0, value=0, step=1, style={"width": "100%"})]),
                        html.Div([html.Label("Holding cost per unit"),
                                  dcc.Input(id="holding-cost", type="number", min=0.01, value=1, step=0.25, style={"width": "100%"})]),
                        html.Div([html.Label("Stockout cost per unit"),
                                  dcc.Input(id="stockout-cost", type="number", min=0.01, value=10, step=0.5, style={"width": "100%"})]),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"},
                ),
                html.Div(id="order-recommendation", style={"marginTop": "16px", "fontSize": "16px", "fontWeight": "600"}),
                html.Div(id="order-method", style={"marginTop": "5px", "fontSize": "12px", "color": "#64748B"}),
            ],
            style={"background": "white", "padding": "18px", "borderRadius": "12px", "marginBottom": "16px"},
        ),

        html.Div(
            [
                html.H3("Forecast Data Preview"),
                dash_table.DataTable(
                    id="forecast-table",
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "fontSize": "12px", "textAlign": "left"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#E2E8F0"},
                ),
                html.Button("Download filtered forecast CSV", id="download-btn",
                            style={"marginTop": "12px", "padding": "10px 15px", "border": "0",
                                   "borderRadius": "7px", "background": BLUE, "color": "white"}),
                dcc.Download(id="download-data"),
            ],
            style={"background": "white", "padding": "18px", "borderRadius": "12px"},
        ),
        html.Div(
            "Metric note: WAPE is shown as a percentage. RMSSE must be calculated using the training-series seasonal-naive scale and exported in model_metrics.csv; it is not inferred from validation data alone.",
            style={"fontSize": "11px", "color": "#64748B", "marginTop": "14px"},
        ),
    ],
)


@app.callback(
    Output("kpi-row", "children"),
    Output("historical-chart", "figure"),
    Output("future-chart", "figure"),
    Output("store-chart", "figure"),
    Output("category-chart", "figure"),
    Output("forecast-table", "data"),
    Output("forecast-table", "columns"),
    Input("state-filter", "value"),
    Input("store-filter", "value"),
    Input("category-filter", "value"),
    Input("item-filter", "value"),
)
def update_dashboard(state, store, category, item):
    h = filter_frame(hist, state, store, item, category)
    f = filter_frame(future, state, store, item, category)

    wape = wape_score(h[ACTUAL_COL], h[PRED_COL]) if not h.empty else np.nan
    rmsse = GLOBAL_RMSSE

    kpis = [
        kpi_card("Filtered WAPE", f"{wape:.2f}%" if pd.notna(wape) else "N/A", "Lower is better"),
        kpi_card("Exported RMSSE", f"{rmsse:.4f}" if pd.notna(rmsse) else "Not supplied", "Requires training scale"),
        kpi_card("Future forecast units", f"{f[FORECAST_COL].sum():,.0f}" if not f.empty else "0", "Sum across filtered horizon"),
        kpi_card("Forecast records", f"{len(f):,}", "Filtered item/store/date rows"),
    ]

    if h.empty:
        hist_fig = go.Figure().add_annotation(text="No historical rows for these filters", showarrow=False)
    else:
        hs = h.groupby(DATE_COL, as_index=False)[[ACTUAL_COL, PRED_COL]].sum().sort_values(DATE_COL)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=hs[DATE_COL], y=hs[ACTUAL_COL], name="Actual", mode="lines"))
        hist_fig.add_trace(go.Scatter(x=hs[DATE_COL], y=hs[PRED_COL], name="Predicted", mode="lines"))
        hist_fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Sales units")

    if f.empty:
        future_fig = go.Figure().add_annotation(text="No future forecast rows for these filters", showarrow=False)
    else:
        fs = f.groupby(DATE_COL, as_index=False)[[FORECAST_COL, "lower", "upper"]].sum(min_count=1).sort_values(DATE_COL)
        future_fig = go.Figure()
        if fs["upper"].notna().any() and fs["lower"].notna().any():
            future_fig.add_trace(go.Scatter(x=fs[DATE_COL], y=fs["upper"], line={"width": 0}, showlegend=False, name="Upper bound"))
            future_fig.add_trace(go.Scatter(x=fs[DATE_COL], y=fs["lower"], fill="tonexty", line={"width": 0},
                                            fillcolor="rgba(37,99,235,0.15)", name="Prediction interval"))
        future_fig.add_trace(go.Scatter(x=fs[DATE_COL], y=fs[FORECAST_COL], name="Forecast", mode="lines+markers"))
        future_fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Forecast sales units")

    if f.empty:
        store_fig = go.Figure()
        cat_fig = go.Figure()
    else:
        by_store = f.groupby("store_id", as_index=False)[FORECAST_COL].sum().sort_values(FORECAST_COL, ascending=False).head(15)
        by_cat = f.groupby("cat_id", as_index=False)[FORECAST_COL].sum().sort_values(FORECAST_COL, ascending=False)
        store_fig = px.bar(by_store, x="store_id", y=FORECAST_COL, template="plotly_white")
        cat_fig = px.bar(by_cat, x="cat_id", y=FORECAST_COL, template="plotly_white")
        store_fig.update_layout(xaxis_title="Store", yaxis_title="Forecast units")
        cat_fig.update_layout(xaxis_title="Category", yaxis_title="Forecast units")

    table_cols = [c for c in [DATE_COL, "state_id", "store_id", "cat_id", "dept_id", "item_id", FORECAST_COL, "lower", "upper"] if c in f.columns]
    table = f[table_cols].copy().sort_values(DATE_COL).head(500) if not f.empty else pd.DataFrame(columns=table_cols)
    if not table.empty:
        table[DATE_COL] = table[DATE_COL].dt.strftime("%Y-%m-%d")
    return kpis, hist_fig, future_fig, store_fig, cat_fig, table.to_dict("records"), [{"name": c, "id": c} for c in table_cols]


@app.callback(
    Output("order-recommendation", "children"),
    Output("order-method", "children"),
    Input("state-filter", "value"),
    Input("store-filter", "value"),
    Input("category-filter", "value"),
    Input("item-filter", "value"),
    Input("on-hand", "value"),
    Input("holding-cost", "value"),
    Input("stockout-cost", "value"),
)
def update_order(state, store, category, item, on_hand, holding_cost, stockout_cost):
    f = filter_frame(future, state, store, item, category)
    if f.empty:
        return "No forecast available for the selected filters.", ""
    demand = f[FORECAST_COL].sum()
    q, method = recommend_order_quantity(
        demand,
        on_hand or 0,
        holding_cost or 1,
        stockout_cost or 10,
    )
    return (
        f"Recommended order quantity: {q:,} units  |  Forecast demand: {demand:,.0f} units",
        f"Optimization method: {method}. This is an aggregate recommendation for the current filters.",
    )


@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("state-filter", "value"),
    State("store-filter", "value"),
    State("category-filter", "value"),
    State("item-filter", "value"),
    prevent_initial_call=True,
)
def download_forecast(n, state, store, category, item):
    f = filter_frame(future, state, store, item, category)
    return dcc.send_data_frame(f.to_csv, "walmart_m5_filtered_forecasts.csv", index=False)


if __name__ == "__main__":
    # Colab: host 0.0.0.0 is important for tunnel access.
    app.run(host="0.0.0.0", port=8050, debug=False)
