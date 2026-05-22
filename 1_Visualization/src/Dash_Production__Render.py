#!/usr/bin/env python
# coding: utf-8

# In[1]:


#basics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#itertools
from itertools import product

# plotly and dash
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import plotly.offline as pyo
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px

#others
from pathlib import Path
import re
import os


# In[2]:


#this notebook is to make the dash plot for the production data and the world map
# I have called modules from the load clean production data and find load plot xy


# In[3]:



# In[4]:


# load the production data 
from Load_clean_production_data import Load_clean_production_data
# Finds the root directory (Volve_Project/) dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(BASE_DIR, '0_Volve_dataset', '5_Production_data', 'Volve_production_data.xlsx')

def load_data():

    return pd.read_excel(data_path)

production_df = load_data()


# In[5]:


# change the production_df from metric to field by changing the dataframe
# this only work on this particular production_df dataframe above
from Load_clean_production_data import metric_to_field
production_df_field=pd.DataFrame()
production_df_field=metric_to_field(production_df)
production_df_field



# In[6]:

# load the XY coordinate data
from Find_Load_plot_X_Y_loc import Fetch_XY_location_make_dataframe
# 1. Get the absolute path of the directory 
SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Navigate up to the project root and down into the target folder

ROOT_DIR = SCRIPT_DIR.parent.parent / "0_Volve_dataset" / "13_Well_technical_data"


# 3. Pass the absolute path to your function
df_headers = Fetch_XY_location_make_dataframe(ROOT_DIR)


# In[7]:


# select the columns needed from the df_headers
df_XY=df_headers[['WELL NAME',
       'WELLBORE NAME', 'Surface EW', 'Surface NS', 'Surface Latitude',
       'Surface Longitude', 'Bottom Hole EW', 'Bottom Hole NS']]
df_XY


# In[8]:


# changes the 'Surface Latitude','Surface Longitude' in degrees to  decimal and gives a new column lat_dd and lon_dd
from Find_Load_plot_X_Y_loc import dms_to_dd
from Find_Load_plot_X_Y_loc import DMS_decimal

df_XY=DMS_decimal(df_XY)
df_XY


# In[9]:


production_df_field = production_df_field.drop(columns=[
    col for col in [
        "WELL_BORE_CODE",
        "NPD_WELL_BORE_CODE",
       "NPD_FIELD_CODE",
       "ON_STREAM_HRS",
        "NPD_FIELD_NAME",
        "NPD_FACILITY_CODE",
        "NPD_FACILITY_NAME",
        "NPD_FIELD_CODE",
        'BORE_WI_VOL', 
        'FLOW_KIND',
        'WELL_TYPE'
        ] if col in production_df_field.columns
    ])

production_df_field


# In[ ]:





# In[10]:


# manually add the coordinates from the xy table to the production data
production_df_field["lat_dd"]=None
production_df_field["lon_dd"]=None

production_df_field.loc[production_df['NPD_WELL_BORE_NAME'] == '15/9-F-1 C', 'lat_dd'] = 58.429546
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-1 C', 'lon_dd'] = 1.887518

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-11', 'lat_dd'] = 58.441655
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-11', 'lon_dd'] = 1.887463

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-12', 'lat_dd'] = 58.440708	
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-12', 'lon_dd'] = 1.878978

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-14', 'lat_dd'] = 58.441602	
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-14', 'lon_dd'] = 1.887522

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-15 D', 'lat_dd'] = 58.441585	
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-15 D', 'lon_dd'] = 1.887541

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-4', 'lat_dd'] = 58.441589	
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-4', 'lon_dd'] = 1.887478

production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-5', 'lat_dd'] = 58.441571	
production_df_field.loc[production_df_field['NPD_WELL_BORE_NAME'] == '15/9-F-5', 'lon_dd'] = 1.887497




# In[11]:


production_df_field


# In[12]:


# fetch the polygon and make it into a dataframe
from Find_Load_plot_X_Y_loc import poly
SCRIPT_DIR = Path(__file__).resolve().parent

polygon_file_path = (
    SCRIPT_DIR.parent.parent 
    / "0_Volve_dataset" 
    / "2_Geophysical_Interpretations" 
    / "Fault_polygons" 
    / "2014_Volve_Hugin_Base.dat"
)

df_poly = poly(polygon_file_path)



# In[25]:


app=dash.Dash()
fig_wells = px.scatter_geo(production_df_field,lat="lat_dd",lon="lon_dd",hover_name="NPD_WELL_BORE_NAME")
app.layout=html.Div([
                    # basic header
                     html.Div([html.H1('Production DashBoard')],style={"border": "1px solid black","height": "8vh","textAlign": "center"}),   
                     # Dropdown

                     # Production plot and xy_map div
                     html.Div([
                             # Production plot
                             html.Div([
                                      dcc.Graph(
                                                id="Production_plot",
                                               style={"width": "100%", "height": "100%"}  # this style is for the graph itself
                                               )
                                        ],style={"border": "1px solid gray", "width": "60%", "height": "60vh" }), # this style is for layout of the div

                             # map plot
                             html.Div([
                                      dcc.Graph(id="xy_map",
                                                figure={"data":fig_wells.data,                                                      # the px.scatter_geo(...) already returns a complete Plotly Figure so use.data to return data
                                                      "layout":go.Layout(title={"text": "Map View", "x": 0.5, "xanchor": "center","y": 0.9,"yanchor": "top","font": {"size": 40}}, 
                                                                         hovermode ="closest",  
                                                                         margin={"l": 0, "r": 0, "t": 10, "b": 0}
                                                                        )
                                                       },
                                                style={"width": "100%", "height": "100%"} # this style is for the graph itself
                                               )
                                     ],style={"border": "1px solid gray", "width": "40%", "height": "60vh"}),  

                     ],style={"border": "1px solid gray", "display": "flex","width": "100vw","height": "60vh"}), # this style is for the production plot and xy_map together

                    # div for all output and gauges
                    html.Div([
                            # Gauge
                             html.Div([
                                     html.Div(
                                              dcc.Graph(
                                                       id="Oil_gauge",
                                                       style={"width": "100%", "height": "100%"}
                                                       ),             
                                              style={"border": "1px solid lightgray", "width": "33.33%", "height": "30vh"}
                                             ),
                                     html.Div(
                                            dcc.Graph(
                                                     id="Gas_gauge",
                                                     style={"width": "100%", "height": "100%"}
                                                     ),
                                            style={"border": "1px solid lightgray", "width": "33.33%", "height": "30vh"}                                 
                                             ),
                                    html.Div(
                                            dcc.Graph(
                                                     id="Water_gauge",
                                                     style={"width": "100%", "height": "100%"}
                                                    ),         
                                           style={"border": "1px solid lightgray", "width": "33.33%", "height": "30vh"})
                                     ],

                                   style={"border": "1px solid gray", "width": "60%", "height": "30vh", 'display': 'flex', 'flex-direction': 'row'}),

                            # div for all table
                    html.Div([
                        html.H3("Decline Curve Parameters"),
                        html.Div(id="well_name",style={"marginBottom": "25px"}),
                        dash_table.DataTable(
                            id="production_table",
                            columns=[
                                {"name": "Last 30 day Avg. Oil Production stb", "id": "oil"},
                                {"name": "Last 30 day Avg. Gas Production mscf", "id": "gas"},
                                {"name": "Last 30 day Avg. Water Production stb", "id": "water"},
                                {"name": "Last 30 day Avg. Pres psi", "id": "pres"},
                            ],
                            data=[],
                            style_table={"overflowX": "auto", "width": "100%"},
                            style_cell={"textAlign": "center", "padding": "8px", "fontFamily": "Arial", "fontSize": "16px", "border": "1px solid lightgray"},
                            style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
                            sort_action="native",
                            filter_action="native",
                            page_size=10
                        )
                    ], style={"border": "1px solid gray", "width": "40%", "height": "30vh"}),

                                        ],style={"display": "flex","flexDirection": "row","width": "100vw","height": "30vh"})

                    ],style={"border": "1px solid gray", "width": "100vw", "height": "100vh"})


#--------------------------------------------------------------------------------------------------------------------
# start of making the production plots
#---------------------------------------------------------------------------------------------------------------------
@app.callback(Output("Production_plot", "figure"),
              [Input("xy_map", "selectedData")])
def plot_production(selectedData):

    if not selectedData or not selectedData.get("points"):
        return go.Figure()
    #print(selectedData)

    wells = [p["hovertext"] for p in selectedData["points"]]
    #print(wells)

    #data = []

    for well in wells:
        filter_df_well = production_df_field[production_df_field["NPD_WELL_BORE_NAME"] == well]

        # make traces for oil gas and water
        trace_oil =go.Scatter(x=filter_df_well["DATEPRD"],y=filter_df_well['BORE_OIL_VOL_STB'], mode = "lines", name= "Oil Rate", line={"color":"green"},yaxis="y2")
        trace_gas =go.Scatter(x=filter_df_well["DATEPRD"],y=filter_df_well['BORE_GAS_VOL_MSCF'], mode = "lines", name= "Gas Rate", line_color="Red")
        trace_water =go.Scatter(x=filter_df_well["DATEPRD"],y=filter_df_well['BORE_WAT_VOL_STB'], mode = "lines", name= "Water Rate" ,line={"color":"blue"},yaxis="y2")
        trace_whp =go.Scatter(x=filter_df_well["DATEPRD"],y=filter_df_well['AVG_WHP_P_PSI'], mode = "lines", name= "WHP" ,line={"color":"purple"},yaxis="y2")

        data=[trace_oil,trace_gas,trace_water,trace_whp]

        layout = go.Layout(
                    title = {"text": f"Wellname<br>{well}", "font": {"size": 22}},
                    xaxis = {"title": {"text": "Time (days)", "font": {"color": "black","size": 36}},"tickfont": {"size": 22, "color": "black"}},
                    yaxis=  {"title": {"text": "Gas Rate (mscf)", "font": {"color": "Red","size": 36}},"tickfont": {"size": 22, "color": "black"}},
                    yaxis2={"title": {"text": '<span style="color:green">Oil Rate (stb)</span><br>'
                                              '<span style="color:blue">Water Rate (stb)</span><br>'
                                              '<span style="color:purple">WHP (psi)</span>', 
                                      "font": {"color": "black","size": 36}}, 
                            "tickfont":{"color":"black", "size":22},  
                            "overlaying":"y",
                            "side":"right"},
                    legend = {"x":1.07, "y":1,"font": {"size": 24}},
                    autosize=True,

                    hovermode = "closest"
                    )
        #fig = go.Figure(data=data)
        #fig.update_xaxes(rangeslider_visible=True, type="date")
        fig =go.Figure(data=data, layout=layout)
        fig.update_xaxes(rangeslider_visible=True,type="date")

        return fig
#--------------------------------------------------------------------------------------------------------------------
# start of the gauges
#---------------------------------------------------------------------------------------------------------------------
# Oil gague   
@app.callback(Output("Oil_gauge", "figure"),
              [Input("xy_map", "selectedData")])
def Oil_gauge(selectedData):
    if not selectedData or not selectedData.get("points"):
        return go.Figure()
    # get selected well name
    well = selectedData["points"][0]["hovertext"]

    # filter production data for that well
    filter_df_well = production_df_field[production_df_field["NPD_WELL_BORE_NAME"] == well]
    # calculate value for gauge
    total_oil = filter_df_well["BORE_OIL_VOL_STB"].sum()
    #print(total_oil)

    max_oil = production_df_field.groupby("NPD_WELL_BORE_NAME")["BORE_OIL_VOL_STB"].sum().max()
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        mode="gauge+number",
        value=total_oil,
        title={"text": f"Total Oil Volume stb<br>{well}", "font": {"size": 32}},
        gauge={
            "axis": {"range": [None, max_oil * 1.1], "tickfont": {"size": 22}},
            "bar": {"color": "gray"},
            "steps": [
                {"range": [0, max_oil * 0.33], "color": "red"},
                {"range": [max_oil * 0.33, max_oil * 0.66], "color": "yellow"},
                {"range": [max_oil * 0.66, max_oil], "color": "green"},
                     ],
           'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_oil*1.05}
        }
    ))
    return fig

# Gas gague 
@app.callback(Output("Gas_gauge", "figure"),
              [Input("xy_map", "selectedData")])
def Gas_gauge(selectedData):
    if not selectedData or not selectedData.get("points"):
        return go.Figure()
    # get selected well name
    well = selectedData["points"][0]["hovertext"]

    # filter production data for that well
    filter_df_well = production_df_field[production_df_field["NPD_WELL_BORE_NAME"] == well]
    # calculate value for gauge
    total_gas = filter_df_well["BORE_GAS_VOL_MSCF"].sum()
    #print(total_gas)

    max_gas = production_df_field.groupby("NPD_WELL_BORE_NAME")["BORE_GAS_VOL_MSCF"].sum().max()
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        mode="gauge+number",
        value=total_gas,
        title={"text": f"Total Gas Volume mscf<br>{well}","font": {"size": 32}},
        gauge={
            "axis": {"range": [None, max_gas * 1.0],"tickfont": {"size": 22}},
            "bar": {"color": "gray"},
            "steps": [
                {"range": [0, max_gas * 0.33], "color": "red"},
                {"range": [max_gas * 0.33, max_gas * 0.66], "color": "yellow"},
                {"range": [max_gas * 0.66, max_gas], "color": "green"},
                     ],
           'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_gas*1.0}
        }
    ))
    return fig

# Water gague 
@app.callback(Output("Water_gauge", "figure"),
              [Input("xy_map", "selectedData")])
def Water_gauge(selectedData):
    if not selectedData or not selectedData.get("points"):
        return go.Figure()
    # get selected well name
    well = selectedData["points"][0]["hovertext"]

    # filter production data for that well
    filter_df_well = production_df_field[production_df_field["NPD_WELL_BORE_NAME"] == well]
    # calculate value for gauge
    total_water = filter_df_well["BORE_WAT_VOL_STB"].sum()
    #print(total_water)

    max_water = production_df_field.groupby("NPD_WELL_BORE_NAME")["BORE_WAT_VOL_STB"].sum().max()
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        mode="gauge+number",
        value=total_water,
        title={"text": f"Total Water Volume stb<br>{well}","font": {"size": 32}},
        gauge={
            "axis": {"range": [None, max_water * 1.2],"tickfont": {"size": 22}},
            "bar": {"color": "gray"},
            "steps": [
                {"range": [0, max_water * 0.33], "color": "green"},
                {"range": [max_water * 0.33, max_water * 0.66], "color": "yellow"},
                {"range": [max_water * 0.66, max_water], "color": "red"},
                     ],
           'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5E6}
        }
    ))
    return fig

#--------------------------------------------------------------------------------------------------------------------
# start of the table
#---------------------------------------------------------------------------------------------------------------------
# Oil output text
@app.callback([Output("production_table", "data"),
              Output("well_name", "children")],
              [Input("xy_map", "selectedData"),])
def update_table(selectedData):
    table_records = []
    if selectedData is None or len(selectedData["points"]) == 0:
        return [], ""
    # get selected well name
    well = selectedData["points"][0]["hovertext"]

    # filter production data for that well
    filter_df_well = production_df_field[production_df_field["NPD_WELL_BORE_NAME"] == well]
    # calculate the last 30 days
    last_oil = filter_df_well["BORE_OIL_VOL_STB"].tail(30).fillna(0).mean()
    last_gas = filter_df_well["BORE_GAS_VOL_MSCF"].tail(30).fillna(0).mean()
    last_water = filter_df_well["BORE_WAT_VOL_STB"].tail(30).fillna(0).mean()
    last_pres = filter_df_well["AVG_WHP_P_PSI"].tail(30).fillna(0).mean()

    table_records.append({"oil": round(last_oil,0), "gas": round(last_gas,0), "water": round(last_water,0), "pres": round(last_pres,0) })
    return table_records, well



if __name__ == '__main__':
    app.run()

