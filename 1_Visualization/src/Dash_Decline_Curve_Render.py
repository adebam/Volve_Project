#!/usr/bin/env python
# coding: utf-8

# In[1]:


#In this notebook, I am going to make a dashboard for Decline curve analysis for the Volve Dataset


# In[1]:


#basics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plotly and dash
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import plotly.graph_objects as go
import traceback

#others
from pathlib import Path
import re
import os


# In[2]:


#import sys
#sys.path.append("../src")   # VERY IMPORTANT


# ## Loading and Processing production data

# In[3]:


from Load_clean_production_data import Load_clean_production_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(BASE_DIR, '0_Volve_dataset', '5_Production_data', 'Volve_production_data.xlsx')

def load_data():

    return pd.read_excel(data_path)

production_df = load_data()
from Load_clean_production_data import metric_to_field
production_df_field=pd.DataFrame()
production_df_field=metric_to_field(production_df)


# In[4]:


# drop columns u dont need
production_df_field = production_df_field.drop(columns=[
    col for col in [
        "WELL_BORE_CODE","NPD_WELL_BORE_CODE", "NPD_FIELD_CODE","ON_STREAM_HRS","NPD_FIELD_NAME","NPD_FACILITY_CODE","NPD_FACILITY_NAME","NPD_FIELD_CODE",'BORE_WI_VOL', 'FLOW_KIND',
        'WELL_TYPE', 'AVG_DOWNHOLE_PRESSURE_PSI','AVG_DOWNHOLE_TEMPERATURE_F', 'AVG_DP_TUBING_PSI','AVG_ANNULUS_PRESS_PSI','AVG_WHP_P_PSI', 'AVG_WHT_P'
        ] if col in production_df_field.columns
    ])


# In[5]:


# calculate running time and normalize time
from Load_clean_production_data import times_calc
production_df_field=times_calc(production_df_field)


# In[6]:


production_df_field


# In[7]:


#
def forecast_dates(df):
    """
    The function takes a dataframe  calculate the last date of production as well as the last days on
    Calculates the calendar days
    Calculates the time production starts, t_forecast
    returns the fture calendar days as well as the future forecast dates as an array
    The maximum prediction days is 15 years this includes the time the well has been on production
    """

    total_predict_days=5475 # 15 years
    last_date = df["DATEPRD"].max()
    last_days_on = df["Days_on_prod"].max()

    calendar_days = (df["DATEPRD"].max()-df["DATEPRD"].min()).days # days on production including when the well is down
    calendar_days = max(calendar_days, 1)
    # forecast producing days
    uptime_fraction = last_days_on / calendar_days
    remaining_calendar_days = max(total_predict_days - calendar_days, 0)
    t_forecast = np.arange(last_days_on,last_days_on + remaining_calendar_days * uptime_fraction,30)  # I want to start the forecast after (5475-normalized_days) 5475 is 15 years

    # estimate uptime
    uptime_fraction = (last_days_on /calendar_days)

    # convert to calendar days
    future_calendar_days = (t_forecast - last_days_on) / uptime_fraction

    # forecast dates
    forecast_dates = (last_date+pd.to_timedelta(future_calendar_days, unit="D"))

    return (future_calendar_days, forecast_dates)


# In[8]:


decline_results = pd.DataFrame({
    "Well": ["15/9-F-1", "15/9-F-11", "15/9-F-12"],
    "Model": ["Exponential", "Hyperbolic", "Hyperbolic"],
    "qi_STB_day": [5200, 4300, 6100],
    "Di_1_day": [0.0012, 0.0009, 0.0015],
    "b_factor": [0.0, 0.85, 1.1],
    "RMSE": [45.2, 62.1, 51.7],
    "EUR_MSTB": [1850, 2400, 3100]
})


# In[9]:


# imports all module for the decline curve analysis
from Decline_curve import get_arps_initial_guess # for initial guesses
from Decline_curve import sort_remove_nan # clean dataframe
from Decline_curve import Exponential # exponential equation
from Decline_curve import Exponential_residuals_log_lmfit # loss function
from Decline_curve import Exponential_minimizer # minimizer

from Decline_curve import Hyperbolic # exponential equation
from Decline_curve import Hyperbolic_residuals_log_lmfit # loss function
from Decline_curve import Hyperbolic_minimizer # minimizer

from Decline_curve import Harmonic # exponential equation
from Decline_curve import Harmonic_residuals_log_lmfit # loss function
from Decline_curve import Harmonic_minimizer # minimizer

from Decline_curve import forecast_dates # minimizer


# ## DASH

# In[ ]:





# In[12]:


app = dash.Dash(__name__)

wellnames = production_df_field["NPD_WELL_BORE_NAME"].unique()
modelnames = ["Exponential Decline", "Hyperbolic Decline", "Harmonic Decline"]

phase_labels = {
    "BORE_OIL_VOL_STB": "Oil Phase, STB/day",
    "BORE_GAS_VOL_MSCF": "Gas Phase, MSCF/day",
    "BORE_WAT_VOL_STB": "Water Phase, STB/day"
}

color_labels = {
    "BORE_OIL_VOL_STB": "green",
    "BORE_GAS_VOL_MSCF": "red",
    "BORE_WAT_VOL_STB": "blue"
}
eur_unit_labels = {
    "BORE_OIL_VOL_STB": "STB",
    "BORE_GAS_VOL_MSCF":"MSCF",
    "BORE_WAT_VOL_STB": "STB"
}

# --- Layout Configuration ---
app.layout = html.Div([
    html.Div([html.H1('Decline Curve Analysis')], style={"border": "1px solid black", "height": "4vh", "textAlign": "center"}),

    html.Div([ 
        html.Div([
            dcc.Graph(
                id="decline_curve", 
                style={"height": "100%", "width": "100%"},
                config={"modeBarButtonsToAdd": ["select2d", "lasso2d"]}
            ),
            dcc.Store(id="stored_selection", storage_type="memory"),
            dcc.Store(id="decline-params-store", storage_type="memory"),
        ], style={"width": "85%", "height": "100%"}), 

        html.Div([
            html.Div([
                html.H4("Select Wells"),
                dcc.Checklist(id="well_checklist", options=wellnames, value=[wellnames[0]] if len(wellnames) > 0 else []),
            ], style={"border": "1px solid lightgray", "width": "100%", "height": "30%"}),

            html.Div([
                html.H4("Select Phase"),
                dcc.RadioItems(
                    id="Phase_selector",
                    options=[
                        {"label": "Oil", "value": "BORE_OIL_VOL_STB"},
                        {"label": "Gas", "value": "BORE_GAS_VOL_MSCF"},
                        {"label": "Water", "value": "BORE_WAT_VOL_STB"}
                    ],
                    value="BORE_OIL_VOL_STB",
                ),
            ], style={"border": "1px solid lightgray", "width": "100%", "height": "30%"}),

            html.Div([
                html.H4("Select Model"),
                dcc.Checklist(id="model_checklist", options=modelnames, value=[]),
            ], style={"border": "1px solid lightgray", "width": "100%", "height": "30%"}),

            html.Div([
                html.H4("Forecast Production"),
                dcc.Checklist(id="forecast_production", options=["Forecast"], value=[]),
            ], style={"border": "1px solid lightgray", "width": "100%", "height": "10%"}),

        ], style={"display": "flex", "flexDirection": "column", "height": "100%", "width": "15%", "border": "solid lightgray"}),

    ], style={"display": "flex", "width": "100vw", "height": "80vh", "border": "1px solid gray"}),

    html.Div([
        html.H3("Decline Curve Parameters"),
        dash_table.DataTable(
            id="dca_parameter_table",
            columns=[
                {"name": "Model", "id": "Model"},
                {"name": "Initial Rate (qi)", "id": "qi"},
                {"name": "Decline Rate (Di)", "id": "Di"},
                {"name": "b-factor (b)", "id": "b"},
                {"name": "EUR (MMSTB) or (Bscf)", "id": "EUR"}
            ],
            data=[],
            style_table={"overflowX": "auto", "width": "100%"},
            style_cell={"textAlign": "center", "padding": "8px", "fontFamily": "Arial", "fontSize": "14px", "border": "1px solid lightgray"},
            style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
            sort_action="native",
            filter_action="native",
            page_size=10
        )
    ], style={"border": "1px solid gray", "width": "100vw", "height": "15vh"}),
], style={"border": "1px solid gray", "width": "100vw", "height": "100vh"})


# ====================================================================================================================
# CALLBACK 1: Cache data selections safely
# ====================================================================================================================
@app.callback(
    Output("stored_selection", "data"),
    Input("decline_curve", "selectedData"),
    State("stored_selection", "data")
)
def store_selection(graph_selection, currently_stored):
    if graph_selection is None or "points" not in graph_selection or len(graph_selection["points"]) == 0:
        return currently_stored if currently_stored else no_update
    return graph_selection


# ====================================================================================================================
# UNIFIED CALLBACK 2: Renders Figure, Fits Arps Models, Generates Line Overlays, Forecasts & Table Rows
# ====================================================================================================================
@app.callback(
    [Output("decline-params-store", "data"),
     Output("decline_curve", "figure"),
     Output("dca_parameter_table", "data")],
    [Input("well_checklist", "value"),
     Input("Phase_selector", "value"),
     Input("model_checklist", "value"),
     Input("stored_selection", "data"),
     Input("forecast_production", "value")] 
)
def update_plot_and_table(selected_well, phase, selected_models, stored_data, forecast_checklist):
    try:
        selected_models = selected_models or []
        forecast_checklist = forecast_checklist or []

        fig = go.Figure()
        dca_params, table_records = {}, []

        if not selected_well:
            fig.update_layout(title={"text": "Select a well to begin"})
            return None, fig, []

        filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_well)].copy().sort_values(by="DATEPRD")
        phase_text = phase_labels.get(phase, phase)
        color = color_labels.get(phase, "black")


        has_selection = stored_data is not None and "points" in stored_data and len(stored_data["points"]) >= 3

        base_opacity = 0.3 if has_selection else 0.9
        fig.add_trace(go.Scatter(x=filtered_df["DATEPRD"], y=filtered_df[phase], mode="markers", name="Production Data", marker={"color": color, "opacity": base_opacity}))

        if has_selection:
            selected_indices = [point["pointIndex"] for point in stored_data["points"]]
            fit_df = filtered_df.iloc[selected_indices].copy().sort_values(by="DATEPRD")

            fig.add_trace(go.Scatter(x=fit_df["DATEPRD"], y=fit_df[phase], mode="markers", name="Selected Points", marker={"color": color, "size": 10, "line": {"width": 1, "color": "black"}}))

            t = fit_df["Days_on_prod"]
            q = fit_df[phase]

            t_forecast_days, t_forecast_dates = forecast_dates(fit_df) 

            # Convert future calendar days back to producing days for the equations! ---
            last_days_on = t.max()
            calendar_days = max((fit_df["DATEPRD"].max() - fit_df["DATEPRD"].min()).days, 1)
            uptime = last_days_on / calendar_days
            t_producing_future = last_days_on + (t_forecast_days * uptime)

            t_end = np.max(t_producing_future) if len(t_producing_future) > 0 else np.max(t)

            qi_exp, Di_exp, qi_hyper, Di_hyper, b_hyper, qi_har, Di_har = [None]*7

            for model in selected_models:
                try:
                    if model == "Exponential Decline":
                        qi_exp, Di_exp = Exponential_minimizer(t, q)
                        q_model_exp = qi_exp * np.exp(-Di_exp * t)
                        fig.add_trace(go.Scatter(x=fit_df["DATEPRD"], y=q_model_exp, mode="lines", name=f"Fit: {model}", line={"width": 3, "color": "orange"}))

                        try:
                            eur_exp = ((qi_exp - (qi_exp * np.exp(-Di_exp * t_end))) / Di_exp)/1e6

                        except: eur_exp = 0

                        table_records.append({"Model": "Exponential", "qi": round(qi_exp, 2), "Di": round(Di_exp, 5), "b": "N/A", "EUR": round(eur_exp, 2)})

                        if "Forecast" in forecast_checklist:
                            q_forecast_exp = qi_exp * np.exp(-Di_exp * t_producing_future)
                            fig.add_trace(go.Scatter(x=t_forecast_dates, y=q_forecast_exp, mode="lines", name=f"Forecast (Exp)", line={"dash": "dash", "width": 3, "color": "orange"}))

                    elif model == "Hyperbolic Decline":
                        qi_hyper, Di_hyper, b_hyper = Hyperbolic_minimizer(t, q)
                        q_model_hyper = qi_hyper / (1 + b_hyper * Di_hyper * t) ** (1 / b_hyper)
                        fig.add_trace(go.Scatter(x=fit_df["DATEPRD"], y=q_model_hyper, mode="lines", name=f"Fit: {model}", line={"width": 3, "color": "red"}))

                        try:
                            q_end = qi_hyper / (1 + b_hyper * Di_hyper * t_end) ** (1 / b_hyper)
                            eur_hyper = ((qi_hyper**b_hyper / (Di_hyper * (1 - b_hyper))) * (qi_hyper**(1 - b_hyper) - q_end**(1 - b_hyper)) if round(b_hyper,3) != 1.0 else (qi_hyper / Di_hyper) * np.log(qi_hyper / q_end))/1e6
                        except: eur_hyper = 0

                        table_records.append({"Model": "Hyperbolic", "qi": round(qi_hyper, 2), "Di": round(Di_hyper, 5), "b": round(b_hyper, 3), "EUR": round(eur_hyper, 2)})

                        if "Forecast" in forecast_checklist:
                            q_forecast_hyper = qi_hyper / (1 + b_hyper * Di_hyper * t_producing_future) ** (1 / b_hyper)
                            fig.add_trace(go.Scatter(x=t_forecast_dates, y=q_forecast_hyper, mode="lines", name=f"Forecast (Hyper)", line={"dash": "dash", "width": 3, "color": "red"}))

                    elif model == "Harmonic Decline":
                        qi_har, Di_har = Harmonic_minimizer(t, q) 
                        q_model_har = qi_har / (1 + Di_har * t) 
                        fig.add_trace(go.Scatter(x=fit_df["DATEPRD"], y=q_model_har, mode="lines", name=f"Fit: {model}", line={"width": 3, "color": "purple"}))

                        try:
                            eur_har = ((qi_har / Di_har) * np.log(qi_har / (qi_har / (1 + Di_har * t_end))))/1e6
                        except: eur_har = 0

                        table_records.append({"Model": "Harmonic", "qi": round(qi_har, 2), "Di": round(Di_har, 5), "b": 1.0, "EUR": round(eur_har, 2)})

                        if "Forecast" in forecast_checklist:
                            q_forecast = qi_har / (1 + Di_har * t_producing_future)
                            fig.add_trace(go.Scatter(x=t_forecast_dates, y=q_forecast, mode="lines", name=f"Forecast (Harmonic)", line={"dash": "dash", "width": 3, "color": "purple"}))

                except Exception as e:
                    print(f"Convergence/Math failed for {model}: {e}")
                    continue

            dca_params = {'qi_exp': qi_exp, 'Di_exp': Di_exp, 'qi_hyper': qi_hyper, 'Di_hyper': Di_hyper, 'b_hyper': b_hyper, 'qi_har': qi_har, 'Di_har': Di_har}

        well_text = ", ".join(selected_well) if isinstance(selected_well, list) else str(selected_well)
        fig.update_layout(
            title={"text": f"{well_text} - {phase_text}", "font": {"size": 24}},
            xaxis_title={"text": "Date", "font": {"size": 24}},
            yaxis_title={"text": phase_text, "font": {"size": 24}},
            hovermode="closest", plot_bgcolor="white", uirevision=well_text
        )

        return dca_params, fig, table_records

    except Exception as e:
        print("\n\n--- CRITICAL CALLBACK ERROR ---")
        traceback.print_exc() 
        return dash.no_update, dash.no_update, dash.no_update 

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




