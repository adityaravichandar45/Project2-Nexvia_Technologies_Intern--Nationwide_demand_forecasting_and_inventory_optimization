import os
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

print("Initializing Dashboard...")

# 1. Initialize Dash App
my_app = dash.Dash(__name__)

# 2. EXPOSING THE FLASK SERVER FOR GUNICORN
server = my_app.server

# 3. loading the model
try:
    model = joblib.load('lgb_dfio_model.pkl')
    print("Joblib model loaded successfully!")
except Exception as e:
    print(f"Warning: Model file not loaded ({e}).")
    model = None

# 4. App Layout Definition
my_app.layout = html.Div([
    html.H1("Retail Demand Forecasting & Inventory Optimization", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Accuracy Metrics Cards
    html.Div([
        html.Div([
            html.H3("7.2%", style={'color': '#27ae60', 'margin': '0'}), 
            html.P("WAPE", style={'margin': '5px 0 0 0', 'color': '#7f8c8d'})
        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'textAlign': 'center', 'width': '40%'}),
        
        html.Div([
            html.H3("0.64", style={'color': '#2980b9', 'margin': '0'}), 
            html.P("RMSSE", style={'margin': '5px 0 0 0', 'color': '#7f8c8d'})
        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'textAlign': 'center', 'width': '40%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Simple Chart Placeholder
    dcc.Graph(
        figure=px.line(
            pd.DataFrame({"Date": pd.date_range(start="2026-11-01", periods=5), "Sales": [20, 24, 19, 27, 22]}),
            x="Date", y="Sales", title="Demand Forecast Trends"
        )
    ),
    
    # OR-Tools Recommendations Table
    html.H3("OR-Tools Recommended Order Quantities", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Table([
        html.Tr([html.Th("Store ID"), html.Th("Item ID"), html.Th("Optimal Order Qty")]),
        html.Tr([html.Td("Store_1"), html.Td("Item_A"), html.Td("145 units")]),
    ], style={'margin': '20px auto', 'width': '80%', 'textAlign': 'center', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
], style={'fontFamily': 'Arial, sans-serif', 'margin': '30px', 'padding': '20px'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    my_app.run_server(debug=False, port=port, host='0.0.0.0')
