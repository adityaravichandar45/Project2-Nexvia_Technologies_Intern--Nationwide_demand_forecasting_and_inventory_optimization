from dash import Dash, html, dcc, callback, Output, Input, State
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Initializing Dash App
app = Dash(__name__)
server = app.server  # Required for cloud deployment


# 2. Loading the model
try:
    model = joblib.load('lgb_dfio_model.pkl')
    print("Joblib model loaded successfully!")
except Exception as e:
    print(f"Warning: Model file not loaded ({e}).")
    model = None

# 2. Defining the 18 feature names required by your model
FEATURE_NAMES = [
    'sell_price', 'year', 'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28'
]


# 4. App Layout
app.layout = html.Div(style={'padding': '30px', 'fontFamily': 'Arial'}, children=[
    html.H1("Demand_Forecasting_and_Inventory_Optimization_Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("Enter feature values below to generate a real-time prediction.", style={'textAlign': 'center'}),

    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '15px', 'marginBottom': '20px'}, children=[
        html.Div([
            html.Label(feature),
            dcc.Input(id=f'input-{feature}', type='number', value=0, style={'width': '100%'})
        ]) for feature in FEATURE_NAMES
    ]),

    html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
        html.Button('Predict Sales', id='button-predict', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
    ]),

    html.Div(id='output-prediction-container', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'marginBottom': '20px'}),

    # Graph to visualize prediction context
    dcc.Graph(id='prediction-graph')
])

# 5. Defining Callback for Interactivity & Prediction
@callback(
    [Output('output-prediction-container', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('button-predict', 'n_clicks')],
    [State(f'input-{feature}', 'value') for feature in FEATURE_NAMES]
)
def make_prediction(n_clicks, *feature_values):
    if n_clicks == 0:
        return "Click 'Predict Sales' to see results.", px.bar(title="Awaiting Input...")

    # Creating a DataFrame for matching the model's expected features
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    # Making prediction using the loaded model
    prediction = model.predict(input_df)[0]

    # Formatting the output text
    output_text = f"Predicted Target Value: {prediction:.2f}"

    # Creating a simple visualization of the inputs/prediction
    fig = px.bar(
        x=FEATURE_NAMES,
        y=feature_values,
        labels={'x': 'Features', 'y': 'Input Value'},
        title=f"Feature Contributions / Inputs (Prediction: {prediction:.2f})"
    )

    return output_text, fig

# 6. Running locally
if __name__ == '__main__':
    app.run(debug=True)
