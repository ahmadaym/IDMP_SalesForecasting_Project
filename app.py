# Import necessary libraries for data manipulation, visualization, web app development, and machine learning
import base64
import io
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, callback_context
import dash_bootstrap_components as dbc
import dash
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# Import custom functions for data processing and modeling from local files
from src.data_process.data_processing import (
    initial_eda,
    generate_pre_cleaning_plots,
    generate_advanced_pre_cleaning_plots,
    perform_data_cleaning,
    generate_post_cleaning_plots,
    generate_advanced_post_cleaning_plots,
    preprocess_data
)
from src.models.modeling import get_models, train_selected_model

# Initialize Flask server to be used with Dash
server = Flask(__name__)

# Configure the Dash app with an external stylesheet and enable URL-based page routing
app = Dash(
    __name__, 
    server=server, 
    url_base_pathname='/', 
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LITERA]
)

# Define a custom loading spinner with a message for user feedback during long processes
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Sales Forecasting App</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom loading spinner text */
            .custom-loading .dash-loading {
                position: relative;
            }
            .custom-loading .dash-spinner::before {
                content: "loading...";
                display: block;
                text-align: center;
                margin-bottom: 10px;
                font-weight: bold;
                font-size: 8px;
                color: #000;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Placeholder for models to be initialized later
models = None

# **Page Layouts**
# Upload page layout for file upload functionality
upload_layout = dbc.Card([
    dbc.CardHeader("Upload Your Data", className="bg-primary text-white"),
    dbc.CardBody([
        html.P("Please upload a CSV file containing your sales data (must include 'Item_Outlet_Sales')."),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
            style={
                'width': '100%',
                'height': '80px',
                'lineHeight': '80px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'textAlign': 'center',
                'margin': '10px 0'
            },
            multiple=False # Only a single file can be uploaded at a time
        ),
        html.Div(id='upload-status'),
        html.Br(),
        dbc.Button('Proceed to EDA', id='proceed-to-eda', color='secondary', disabled=True)
    ])
], className="mb-4")

# EDA (Exploratory Data Analysis) page layout
eda_layout = html.Div([
    dbc.Card([
        dbc.CardHeader("Exploratory Data Analysis", className="bg-primary text-white"),
        dbc.CardBody([
            html.Div(id='eda-content'),
            html.Br(),
            dbc.Alert("Once you're done reviewing the EDA, click 'Proceed to Cleaning' below.", color="info"),
            dbc.Button('Proceed to Cleaning', id='proceed-to-cleaning', color='primary', className='mt-3', disabled=True)
        ])
    ])
])

# Data Cleaning page layout
cleaning_layout = html.Div([
    dbc.Card([
        dbc.CardHeader("Data Cleaning", className="bg-primary text-white"),
        dbc.CardBody([
            html.P("Handle missing values, outliers, etc."),
            dbc.Button('Perform Data Cleaning', id='perform-cleaning-button', color='warning'),
            html.Br(), html.Br(),
            html.Div(id='post-cleaning-visuals'),
            html.Br(),
            dbc.Alert("After reviewing the cleaning results, click 'Proceed to Modeling' below.", color="info"),
            dbc.Button('Proceed to Modeling', id='proceed-to-modeling', color='primary', className='mt-3', disabled=True)
        ])
    ])
])

# Model Training page layout
modeling_layout = html.Div([
    dbc.Card([
        dbc.CardHeader("Model Training", className="bg-primary text-white"),
        dbc.CardBody([
            html.P("Train multiple models and compare their performance."),
            dbc.Button('Train All Models', id='train-all-models-button', color='success', disabled=True),
            html.Br(), html.Br(),
            # Custom loading text added via CSS class "custom-loading"
            dcc.Loading(id="loading-training", type="circle", className="custom-loading", children=html.Div(id='all-models-output')),
            html.Br(),
            dbc.Alert("After reviewing your model results, click 'Proceed to Results' below.", color="info"),
            dbc.Button('Proceed to Results', id='proceed-to-results', color='primary', className='mt-3', disabled=True)
        ])
    ])
])

# Results & Downloads page layout
results_layout = html.Div([
    dbc.Card([
        dbc.CardHeader("Results & Downloads", className="bg-primary text-white"),
        dbc.CardBody([
            html.Div(id='results-content'),
            html.Br(),
            dbc.Button("Download Best Model Predictions", id="download-predictions-button", color='info', disabled=True, className="me-2"),
            dcc.Download(id="download-predictions"),
            dbc.Button("Download Summary Metrics", id="download-summary-button", color='info', disabled=True),
            dcc.Download(id="download-summary")
        ])
    ])
], className="mb-4")

# **About Us Modal**
# Pop-up modal with information about the development team
about_modal = dbc.Modal(
    [
        dbc.ModalHeader("About Our Team"),
        dbc.ModalBody(
            "We are Ayman, Amisha, and Ronhit—a dedicated team passionate about data science "
            "and machine learning. Our goal is to help businesses forecast sales, understand "
            "their data, and make data-driven decisions with confidence."
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-about-modal", className="ml-auto", n_clicks=0)
        ),
    ],
    id="about-modal",
    is_open=False,
)

# Main layout that includes a navigation bar, steps, and page layouts
app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='original-data-store'),  # Stores original uploaded data
    dcc.Store(id='cleaned-data-store'),  # Stores cleaned data after processing
    dcc.Store(id='eda-completed', data=False),  # Flags if EDA is completed
    dcc.Store(id='cleaning-completed', data=False),  # Flags if cleaning is completed
    dcc.Store(id='modeling-completed', data=False),  # Flags if modeling is completed
    dcc.Store(id='best-model-name'),  # Stores the name of the best model
    dcc.Store(id='best-model-predictions'),  # Stores predictions of the best model
    dcc.Store(id='summary-metrics'),  # Stores summary metrics for models
    dcc.Store(id='model-results-store'),  # Stores all model results for detailed analysis
    dcc.Location(id='url', refresh=False),  # Tracks URL for navigation between pages

    # Navigation bar with "About Us" link
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Sales Forecasting", className="ms-2", style={'fontSize': '1.5em', 'fontWeight': 'bold'}),
            dbc.NavItem(dbc.NavLink("About Us", href="#", id="open-about-modal", style={'color': 'white'}))
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),

    about_modal, # Include "About Us" modal in layout

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Steps", className="mb-4"),
                dbc.Nav(
                    [
                        dbc.NavLink("1. Upload Data", href="/upload", active="exact", id="nav-upload"),
                        dbc.NavLink("2. EDA", href="/eda", active="exact", id="nav-eda", disabled=True),
                        dbc.NavLink("3. Cleaning", href="/cleaning", active="exact", id="nav-cleaning", disabled=True),
                        dbc.NavLink("4. Modeling", href="/modeling", active="exact", id="nav-modeling", disabled=True),
                        dbc.NavLink("5. Results", href="/results", active="exact", id="nav-results", disabled=True),
                    ],
                    vertical=True,
                    pills=True
                )
            ], style={'position': 'sticky', 'top': '20px'})
        ], width=2),

        dbc.Col([
            html.Div(id='upload-page', children=upload_layout, style={'display': 'block'}),
            html.Div(id='eda-page', children=eda_layout, style={'display': 'none'}),
            html.Div(id='cleaning-page', children=cleaning_layout, style={'display': 'none'}),
            html.Div(id='modeling-page', children=modeling_layout, style={'display': 'none'}),
            html.Div(id='results-page', children=results_layout, style={'display': 'none'})
        ], width=10)
    ], style={'marginBottom': '50px'}),  # Add margin bottom to make space for footer

    # Footer
    html.Footer("Created by Ayman, Amisha & Ronhit.", style={
        'textAlign': 'center', 
        'marginTop': '50px', 
        'fontSize': '0.9em', 
        'color': '#555'
    })
], style={'maxWidth': '95%', 'margin': 'auto'})

# Callback to switch between pages based on URL
@app.callback(
    [Output('upload-page', 'style'),
     Output('eda-page', 'style'),
     Output('cleaning-page', 'style'),
     Output('modeling-page', 'style'),
     Output('results-page', 'style')],
    Input('url', 'pathname')
)
def switch_page_display(pathname):
    # Define the hidden style for inactive pages
    hidden = {'display': 'none'}
    # Check the URL pathname to determine which page to display
    if pathname in [None, '/', '/upload']:
        return {'display': 'block'}, hidden, hidden, hidden, hidden
    elif pathname == '/eda':
        return hidden, {'display': 'block'}, hidden, hidden, hidden
    elif pathname == '/cleaning':
        return hidden, hidden, {'display': 'block'}, hidden, hidden
    elif pathname == '/modeling':
        return hidden, hidden, hidden, {'display': 'block'}, hidden
    elif pathname == '/results':
        return hidden, hidden, hidden, hidden, {'display': 'block'}

    # Default case - show the upload page
    return {'display': 'block'}, hidden, hidden, hidden, hidden

# Callback to enable/disable navigation links between the steps
@app.callback(
    [Output('nav-eda', 'disabled'),
     Output('nav-cleaning', 'disabled'),
     Output('nav-modeling', 'disabled'),
     Output('nav-results', 'disabled')],
    [Input('original-data-store', 'data'),
     Input('eda-completed', 'data'),
     Input('cleaning-completed', 'data'),
     Input('modeling-completed', 'data')]
)

def update_nav_links(original_data, eda_done, cleaning_done, modeling_done):
    # Control which steps are clickable based on completion of previous steps
    eda_disabled = not bool(original_data)  # EDA is available after data is uploaded
    cleaning_disabled = not bool(eda_done)  # Cleaning is available after EDA is complete
    modeling_disabled = not bool(cleaning_done)  # Modeling is available after cleaning is complete
    results_disabled = not bool(modeling_done)  # Results are available after modeling is complete
    return eda_disabled, cleaning_disabled, modeling_disabled, results_disabled

# Callback to handle file upload and update the store with the uploaded data
@app.callback(
    [Output('original-data-store', 'data'),
     Output('upload-status', 'children'),
     Output('proceed-to-eda', 'disabled')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def handle_file_upload(contents, filename):
    if contents is not None:
        # Split the contents to retrieve the file data
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string) # Decode the base64 encoded data
        try:
            # Check if the file is a CSV
            if 'csv' in filename.lower():
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))  # Convert file contents to DataFrame
                if 'Item_Outlet_Sales' not in df.columns:
                    return None, dbc.Alert("The CSV must contain 'Item_Outlet_Sales' column.", color="danger"), True
                
                # Check if dataset has time column
                has_time = 'Time' in df.columns
                global models
                models = get_models(has_time=has_time) # Load models based on dataset type

                status = dbc.Alert(
                    "Data uploaded successfully! Detected " +
                    ("time-series data." if has_time else "standard data.") +
                    " Click 'Proceed to EDA' to continue.",
                    color="success"
                )
                return df.to_dict('records'), status, False  # Update the store with data and enable EDA button
            else:
                return None, dbc.Alert("Unsupported file format. Please upload a CSV file.", color="danger"), True
        except Exception as e:
            return None, dbc.Alert(f"Error processing the file: {e}", color="danger"), True
    else:
        return None, dbc.Alert("No data uploaded yet.", color="info"), True

# Callback to display the EDA content once data is uploaded
@app.callback(
    Output('eda-content', 'children'),
    Input('original-data-store', 'data')
)
def show_eda(original_data):
    if original_data is not None:
        df = pd.DataFrame(original_data)  # Convert stored data to DataFrame
        eda_results = initial_eda(df)  # Generate basic EDA results
        initial_plots = generate_pre_cleaning_plots(df)  # Generate basic visualizations
        advanced_plots = generate_advanced_pre_cleaning_plots(df)  # Generate advanced EDA visualizations

        layout = [
            html.H5("Data Overview"),
            eda_results, # Display tabular overview of the data
            html.Hr(),
            html.H5("Initial Exploratory Visualizations"),
            html.Div(initial_plots), # Display plots for exploratory analysis
            html.H5("Advanced EDA"),
            html.Div(advanced_plots), # Display advanced visualizations
        ]
        return layout # Return the layout to be displayed on the page
    else:
        return "Please upload data first."

# Callback to enable the button for proceeding to data cleaning after EDA
@app.callback(
    Output('proceed-to-cleaning', 'disabled'),
    Input('proceed-to-eda', 'n_clicks'),
    State('original-data-store', 'data')
)
def enable_proceed_to_cleaning(n_clicks, original_data):
    if n_clicks and original_data is not None:
        return False  # Enable the button to move to cleaning if EDA is complete
    return True

# Callback to mark EDA as completed once the button is clicked
@app.callback(
    Output('eda-completed', 'data'),
    Input('proceed-to-cleaning', 'n_clicks'),
    State('original-data-store', 'data'),
    prevent_initial_call=True # Only run this function if button is clicked
)
def mark_eda_done(n_clicks, original_data):
    if n_clicks and original_data is not None:
        return True # Indicate that EDA is complete
    return no_update

# Callback to show post-cleaning visuals
@app.callback(
    Output('post-cleaning-visuals', 'children'),
    Input('perform-cleaning-button', 'n_clicks'),
    State('original-data-store', 'data')
)
def perform_data_cleaning_and_show_visuals(n_clicks, original_data):
    if n_clicks and original_data is not None:
        df = pd.DataFrame(original_data)  # Convert stored data to DataFrame
        cleaned_df = perform_data_cleaning(df.copy())  # Perform data cleaning
        post_cleaning_plots = generate_post_cleaning_plots(cleaned_df)  # Generate post-cleaning visualizations
        advanced_post_cleaning = generate_advanced_post_cleaning_plots(df, cleaned_df)  # Generate advanced visuals
        visuals = [
            html.H5('Post-Cleaning Visualizations'),
            html.Div(post_cleaning_plots), # Display the visualizations
            html.H5('Additional Post-Cleaning Analysis'),
            html.Div(advanced_post_cleaning) # Display additional visuals
        ]
        return visuals
    else:
        return "No cleaning performed yet."

# Callback to store cleaned data
@app.callback(
    Output('cleaned-data-store', 'data'),
    Input('perform-cleaning-button', 'n_clicks'),
    State('original-data-store', 'data')
)
def store_cleaned_data(n_clicks, original_data):
    if n_clicks and original_data:
        df = pd.DataFrame(original_data)
        cleaned_df = perform_data_cleaning(df.copy())
        return cleaned_df.to_dict('records')  # Store the cleaned data
    return no_update

# Callback to enable the button for proceeding to modeling after cleaning
@app.callback(
    Output('proceed-to-modeling', 'disabled'),
    Input('perform-cleaning-button', 'n_clicks'),
    State('original-data-store', 'data')
)
def enable_modeling(n_clicks, original_data):
    if n_clicks and original_data is not None:
        return False # Enable the button to move to modeling
    return True

# Callback to mark cleaning as completed
@app.callback(
    Output('cleaning-completed', 'data'),
    Input('proceed-to-modeling', 'n_clicks'),
    State('cleaned-data-store', 'data')
)
def mark_cleaning_done(n_clicks, cleaned_data):
    if n_clicks and cleaned_data is not None:
        return True # Indicate that cleaning is complete
    return no_update

# Callback to enable the 'Train All Models' button once cleaning is complete
@app.callback(
    Output('train-all-models-button', 'disabled'),
    Input('proceed-to-modeling', 'n_clicks'),
    State('cleaned-data-store', 'data')
)
def enable_train_button(n_clicks, cleaned_data):
    # Enable the button only if user has completed cleaning and clicks on 'Proceed to Modeling'
    if n_clicks and cleaned_data is not None:
        return False  # Enable the 'Train All Models' button
    return True  # Disable the button until conditions are met

# Callback to train all models and display the results
@app.callback(
    [Output('all-models-output', 'children'),
     Output('best-model-name', 'data'),
     Output('best-model-predictions', 'data'),
     Output('summary-metrics', 'data'),
     Output('download-predictions-button', 'disabled'),
     Output('download-summary-button', 'disabled'),
     Output('model-results-store', 'data'),
     Output('modeling-completed', 'data')],
    Input('train-all-models-button', 'n_clicks'),
    [State('cleaned-data-store', 'data'), State('original-data-store', 'data')]
)
def train_all_models(n_clicks, cleaned_data, original_data):
    if n_clicks and cleaned_data is not None:
        # Convert stored data to DataFrame
        cleaned_sales_data = pd.DataFrame(cleaned_data)
        original_sales_data = pd.DataFrame(original_data)

        # Check if this is a time-series dataset
        is_time_series = 'Time' in original_sales_data.columns

        if is_time_series:
            run_models = {'ARIMA': models['ARIMA']} # Only train ARIMA for time-series
            X_train, X_test, y_train, y_test, feature_names = None, None, None, None, None
        else:
            run_models = models # Train all models for non-time-series
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(cleaned_sales_data)
            # Scale the feature data to ensure better model performance
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

        results_list = [] # Store model results
        for model_name, model_obj in run_models.items():
            results = train_selected_model(
                model_name,
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names,
                is_time_series=is_time_series,
                original_data=original_sales_data if is_time_series else None
            )
            results_list.append((model_name, results)) # Store model name and results

        # Determine best performing model based on R² (Test) score
        best_model_name = None
        best_r2_score = -np.inf
        best_model_predictions = None
        best_model_obj = None

        summary_data = []
        # Identify the best model
        for model_name, res in results_list:
            metrics = res['metrics']
            summary_data.append({
                'Model': model_name,
                'R² (Test)': metrics['r2_test'],
                'RMSE (Test)': metrics['rmse_test'],
                'MAE (Test)': metrics['mae_test'],
                'MAPE (Test)': metrics['mape_test'],
                'CV R² Mean': metrics['cv_r2_mean'],
                'CV RMSE Mean': metrics['cv_rmse_mean'],
                'MedAE (Test)': metrics.get('medae_test', np.nan)
            })
            if metrics['r2_test'] > best_r2_score:
                best_r2_score = metrics['r2_test']
                best_model_name = model_name
                # Load model object
                best_model_obj = res['model']

        if best_model_obj is not None:
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_model_obj, f)

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data) # Create DataFrame of model metrics
        metric_cols = ['R² (Test)', 'RMSE (Test)', 'MAE (Test)', 'MAPE (Test)', 'CV R² Mean', 'CV RMSE Mean', 'MedAE (Test)']
        summary_df[metric_cols] = summary_df[metric_cols].apply(pd.to_numeric, errors='coerce')
        for col in metric_cols:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NA")

        summary_table = dash_table.DataTable(
            data=summary_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in summary_df.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230,230,230)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left'}
        )

        comparison_metrics = ['R² (Test)', 'RMSE (Test)', 'MAE (Test)', 'MAPE (Test)']
        comparison_df = summary_df.melt(id_vars='Model', value_vars=comparison_metrics, var_name='Metric', value_name='Value')

        primary_metrics = ['R² (Test)', 'MAPE (Test)']
        secondary_metrics = ['RMSE (Test)', 'MAE (Test)']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(title='Model Comparison on Key Metrics', barmode='group')

        for metric in primary_metrics:
            sub_df = comparison_df[comparison_df['Metric'] == metric]
            fig.add_trace(
                go.Bar(x=sub_df['Model'], y=sub_df['Value'], name=metric, text=sub_df['Value'], textposition='outside'),
                secondary_y=False
            )

        for metric in secondary_metrics:
            sub_df = comparison_df[comparison_df['Metric'] == metric]
            fig.add_trace(
                go.Bar(x=sub_df['Model'], y=sub_df['Value'], name=metric, text=sub_df['Value'], textposition='outside'),
                secondary_y=True
            )

        fig.update_yaxes(title_text="Primary Metric Scale", secondary_y=False)
        fig.update_yaxes(title_text="Secondary Metric Scale", secondary_y=True)

        cleaned_results = {}
        # We will store model objects temporarily to handle ARIMA predictions after best model is chosen
        model_objects = {}

        for m_name, res in results_list:
            # Keep model object separately
            model_objects[m_name] = res['model']
            # Remove model object from results before storing
            if 'model' in res:
                del res['model']
            fig_keys = ['importance_fig', 'residual_fig', 'pred_vs_actual_fig', 'residuals_vs_fitted_fig', 'shap_fig']
            for fk in fig_keys:
                if fk in res and res[fk] is not None:
                    res[fk] = res[fk].to_dict()

            if 'pdp_figs' in res and res['pdp_figs'] is not None:
                new_pdp = []
                for f in res['pdp_figs']:
                    if f is not None:
                        new_pdp.append(f.to_dict())
                res['pdp_figs'] = new_pdp

            cleaned_results[m_name] = res

        # After identifying best model, generate predictions for the best model
        best_res = [r for (name, r) in results_list if name == best_model_name][0]
        best_model_obj = model_objects[best_model_name]

        if best_model_name == 'ARIMA':
            # For ARIMA, we do not have X_test. Instead, we have y_test_arima in best_res
            y_test_arima_values = best_res['y_test_arima']
            y_test_arima_series = pd.Series(y_test_arima_values)
            preds = best_model_obj.predict(n_periods=len(y_test_arima_series))
            best_model_predictions = pd.DataFrame({'Actual': y_test_arima_series, 'Predicted': preds})
        else:
            # Non-ARIMA
            # We re-run preprocess_data since we need X_test and y_test
            if not is_time_series:
                X_train, X_test, y_train, y_test, feature_names = preprocess_data(pd.DataFrame(cleaned_data))
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
                preds = best_model_obj.predict(X_test)
                best_model_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
            else:
                # If it's time_series but best is not ARIMA, theoretically should not happen since we only train ARIMA in that scenario
                best_model_predictions = pd.DataFrame()

        tabs_list = []
        for model_name in cleaned_results.keys():
            model_res = cleaned_results[model_name]
            display_name = model_name
            if model_name == best_model_name:
                display_name = f"{model_name} ⭐"

            detail_cards = []
            if model_res.get('best_params'):
                detail_cards.append(html.P(f"Best Parameters: {model_res['best_params']}"))

            if model_res.get('importance_fig'):
                detail_cards.append(html.H5("Feature Importance"))
                detail_cards.append(dcc.Graph(figure=model_res['importance_fig']))

            if model_res.get('residual_fig'):
                detail_cards.append(html.H5("Residual Distribution"))
                detail_cards.append(dcc.Graph(figure=model_res['residual_fig']))

            if model_res.get('pred_vs_actual_fig'):
                detail_cards.append(html.H5("Predicted vs Actual"))
                detail_cards.append(dcc.Graph(figure=model_res['pred_vs_actual_fig']))

            if model_res.get('residuals_vs_fitted_fig'):
                detail_cards.append(html.H5("Residuals vs Fitted"))
                detail_cards.append(dcc.Graph(figure=model_res['residuals_vs_fitted_fig']))

            if model_res.get('pdp_figs'):
                for i, pdp_fig in enumerate(model_res['pdp_figs']):
                    detail_cards.append(html.H5(f"PDP Plot {i+1}"))
                    detail_cards.append(dcc.Graph(figure=pdp_fig))

            if model_res.get('shap_fig'):
                detail_cards.append(html.H5("SHAP Feature Importance"))
                detail_cards.append(dcc.Graph(figure=model_res['shap_fig']))

            if not detail_cards:
                detail_cards.append(html.P("No additional details available."))

            tabs_list.append(
                dbc.Tab(label=display_name, tab_id=model_name, children=detail_cards)
            )

        tabs = dbc.Tabs(id="models-tabs", active_tab=best_model_name, children=tabs_list)

        final_layout = [
            html.H3('All Models Trained Successfully', className='text-success'),
            html.H4(f'Best Performing Model: {best_model_name} ⭐', className='mt-3', style={'color': '#d9534f', 'fontWeight': 'bold'}),
            html.H4('Summary of Results', className='mt-4'),
            summary_table,
            html.H4("Model Comparison on Key Metrics", className='mt-4'),
            dcc.Graph(figure=fig),
            html.H4("Detailed Analysis of Each Model", className='mt-4'),
            tabs
        ]

        return (final_layout,
                best_model_name,
                best_model_predictions.to_dict('records') if best_model_predictions is not None else [],
                summary_df.to_dict('records'),
                False,
                False,
                cleaned_results,
                True)
    elif n_clicks:
        return (html.Div(['Please complete cleaning before training the models.']),
                no_update, no_update, no_update, True, True, no_update, no_update)
    else:
        return html.Div(), no_update, no_update, no_update, True, True, no_update, no_update

# Callback to display the results summary and best model
@app.callback(
    Output('results-content', 'children'),
    Input('modeling-completed', 'data'),
    State('best-model-name', 'data'),
    State('summary-metrics', 'data')
)
def show_results(modeling_done, best_model, summary_data):
    if modeling_done and best_model and summary_data:
        return [
            html.H3("Final Results", className='text-success'),
            html.P(f"The best model is: {best_model}"),
            html.P("You can now download the predictions and summary metrics below.")
        ]
    return "Please train models first."

@app.callback(
    Output("download-predictions", "data"),
    Input("download-predictions-button", "n_clicks"),
    State("best-model-predictions", "data"),
    prevent_initial_call=True
)
def download_predictions(n_clicks, predictions_data):
    if predictions_data:
        df = pd.DataFrame(predictions_data)
        return dcc.send_data_frame(df.to_csv, "best_model_predictions.csv")

@app.callback(
    Output("download-summary", "data"),
    Input("download-summary-button", "n_clicks"),
    State("summary-metrics", "data"),
    prevent_initial_call=True
)
def download_summary(n_clicks, summary_data):
    if summary_data:
        df = pd.DataFrame(summary_data)
        return dcc.send_data_frame(df.to_csv, "summary_metrics.csv")

# Callback to enable the button to proceed to the Results page
@app.callback(
    Output('proceed-to-results', 'disabled'),
    Input('modeling-completed', 'data')
)
def enable_proceed_to_results_button(modeling_done):
    if modeling_done:
        return False
    return True

# Callback to handle the navigation between steps of the app
@app.callback(
    Output('url', 'pathname'),
    [Input('proceed-to-eda', 'n_clicks'),
     Input('proceed-to-cleaning', 'n_clicks'),
     Input('proceed-to-modeling', 'n_clicks'),
     Input('proceed-to-results', 'n_clicks')],
    [State('original-data-store', 'data'),
     State('eda-completed', 'data'),
     State('cleaning-completed', 'data'),
     State('modeling-completed', 'data')],
    prevent_initial_call=True # Only run this function if one of the buttons is clicked
)
def navigate_steps(proceed_eda_clicks, proceed_cleaning_clicks, proceed_modeling_clicks, proceed_results_clicks, original_data, eda_done, cleaning_done, modeling_done):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'proceed-to-eda' and original_data is not None:
        return '/eda'
    if triggered_id == 'proceed-to-cleaning' and eda_done:
        return '/cleaning'
    if triggered_id == 'proceed-to-modeling' and cleaning_done:
        return '/modeling'
    if triggered_id == 'proceed-to-results' and modeling_done:
        return '/results'
    return no_update

# Callback to toggle the visibility of the "About Us" modal
@app.callback(
    Output("about-modal", "is_open"),
    [Input("open-about-modal", "n_clicks"), Input("close-about-modal", "n_clicks")],
    [State("about-modal", "is_open")]
)
def toggle_about_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open # Toggle the visibility of the modal
    return is_open

# Entry point for the Dash application
if __name__ == '__main__':
    app.run_server(debug=True) # Run the server in debug mode for live updates
