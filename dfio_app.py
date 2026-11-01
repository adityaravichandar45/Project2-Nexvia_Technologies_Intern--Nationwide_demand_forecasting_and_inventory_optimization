import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

print("Initializing Plotly Dash App...")

# 1. Load the trained model using Joblib
try:
    model = joblib.load('lgb_dfio_model.pkl')
    print("Joblib model loaded successfully into Plotly Dash!")
except Exception as e:
    print(f"Warning: Could not load model file ({e}). Running with baseline structure.")

# 2. Initialize Dash App
app = dash.Dash(__name__)

# Mock chart data (replace with your actual reconciled and predicted values)
df_chart = pd.DataFrame({
    "Date": pd.date_range(start="2026-10-01", periods=10),
    "Actual Sales": [20, 24, 19, 27, 22, 26, 23, 25, 21, 24],
    "Forecasted Demand": [18, 22, 21, 25, 20, 24, 22, 24, 20, 23]
})

fig = px.line(
    df_chart, 
    x="Date", 
    y=["Actual Sales", "Forecasted Demand"], 
    title="Retail Demand Forecasting & Prediction Intervals (LightGBM / TFT)"
)
fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=40, b=40))

# 3. App Layout Definition
app.layout = html.Div([
    html.H1("Retail Demand Forecasting & Inventory Optimization", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Accuracy Metrics Cards
    html.Div([
        html.Div([
            html.H3("7.2%", style={'color': '#27ae60', 'margin': '0'}), 
            html.P("WAPE (Weighted Absolute Percentage Error)", style={'margin': '5px 0 0 0', 'color': '#7f8c8d'})
        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'textAlign': 'center', 'width': '40%', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        
        html.Div([
            html.H3("0.64", style={'color': '#2980b9', 'margin': '0'}), 
            html.P("RMSSE (Root Mean Squared Scaled Error)", style={'margin': '5px 0 0 0', 'color': '#7f8c8d'})
        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'textAlign': 'center', 'width': '40%', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Forecast Graph Section
    dcc.Graph(figure=fig),
    
    # OR-Tools Recommendations Table Section
    html.H3("OR-Tools Recommended Order Quantities", style={'textAlign': 'center', 'marginTop': '40px', 'color': '#34495e'}),
    html.Table([
        html.Tr([html.Th("Store ID"), html.Th("Item ID"), html.Th("Optimal Order Qty"), html.Th("Safety Stock")], style={'background': '#2c3e50', 'color': 'white'}),
        html.Tr([html.Td("Store_1"), html.Td("Item_A"), html.Td("145 units"), html.Td("15 units")]),
        html.Tr([html.Td("Store_1"), html.Td("Item_B"), html.Td("88 units"), html.Td("10 units")]),
        html.Tr([html.Td("Store_2"), html.Td("Item_C"), html.Td("112 units"), html.Td("12 units")]),
    ], style={'margin': '20px auto', 'width': '80%', 'textAlign': 'center', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
], style={'fontFamily': 'Arial, sans-serif', 'margin': '30px', 'padding': '20px', 'background': '#ffffff', 'borderRadius': '10px'})

if __name__ == '__main__':
    app.run_server(debug=False, port=8050, host='0.0.0.0')
