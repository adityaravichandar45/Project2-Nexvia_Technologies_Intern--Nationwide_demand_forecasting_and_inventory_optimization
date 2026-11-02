from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize Dash App
app = Dash(__name__)
server = app.server

# Sample historical / inference data based on the M5 dataset features found in the model
df_sample = pd.DataFrame({
    'item_id': ['HOBBIES_1_001', 'HOBBIES_1_002', 'HOUSEHOLD_1_001', 'FOODS_1_001'],
    'store_id': ['CA_1', 'CA_1', 'TX_1', 'WI_1'],
    'state_id': ['CA', 'CA', 'TX', 'WI'],
    'cat_id': ['HOBBIES', 'HOBBIES', 'HOUSEHOLD', 'FOODS'],
    'dept_id': ['HOBBIES_1', 'HOBBIES_1', 'HOUSEHOLD_1', 'FOODS_1'],
    'sell_price': [2.99, 4.50, 8.99, 1.25],
    'month': [5, 5, 5, 5],
    'year': [2026, 2026, 2026, 2026],
    'weekday': [0, 1, 2, 3],
    'event_name_1': ['None', 'SuperBowl', 'None', 'LaborDay'],
    'predicted_sales': [12.5, 8.2, 15.0, 42.1]
})

app.layout = html.Div(style={'padding': '30px', 'fontFamily': 'sans-serif'}, children=[
    html.H1("Demand Forecasting & Inventory Optimization Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("Interactive dashboard built with Plotly Dash using LightGBM model features (item_id, store_id, sell_price, etc.).", style={'textAlign': 'center', 'color': '#7f8c8d'}),
    
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '30px'}, children=[
        html.Div([
            html.Label("Select State:"),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': s, 'value': s} for s in df_sample['state_id'].unique()],
                value='CA',
                style={'width': '200px'}
            )
        ]),
        html.Div([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id='cat-dropdown',
                options=[{'label': c, 'value': c} for c in df_sample['cat_id'].unique()],
                value='HOBBIES',
                style={'width': '200px'}
            )
        ])
    ]),
    
    dcc.Graph(id='sales-prediction-graph'),
    
    html.Div(style={'marginTop': '40px'}, children=[
        html.H3("Model Feature Summary"),
        html.Ul([
            html.Li("Features: item_id, store_id, state_id, cat_id, dept_id, sell_price"),
            html.Li("Temporal Features: month, year, weekday, event_name_1"),
            html.Li("Lag & Rolling Features: lag_1, lag_7, rolling_mean_7, rolling_std_7, etc.")
        ])
    ])
])

@callback(
    Output('sales-prediction-graph', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('cat-dropdown', 'value')]
)
def update_dashboard(selected_state, selected_cat):
    filtered_df = df_sample[(df_sample['state_id'] == selected_state) & (df_sample['cat_id'] == selected_cat)]
    if filtered_df.empty:
        filtered_df = df_sample # Fallback
        
    fig = px.bar(
        filtered_df, 
        x='item_id', 
        y='predicted_sales', 
        color='store_id',
        title=f'Predicted Sales for {selected_cat} in {selected_state}',
        labels={'item_id': 'Item ID', 'predicted_sales': 'Predicted Sales (Units)'}
    )
    fig.update_layout(template='plotly_white')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
