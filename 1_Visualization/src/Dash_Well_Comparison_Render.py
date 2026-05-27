#!/usr/bin/env python
# coding: utf-8

# In[1]:


#basics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#itertools
from itertools import product

# plotly and dash
import dash
from dash import dcc, html
import plotly.offline as pyo
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
#import dash_bootstrap_components as dbc

#others
from pathlib import Path
import re
import os
#import flask


# In[2]:


# in this project, I am going to load the grid and petrophysical data and also load the production data so you can compare production of wells to eachother
# I will also change make sure I can change the y and x axis


# In[3]:


# ## Processing and Loading Petrophysical data & Well spec data from the eclipse file

# In[4]:


# Import all petrophysical data as a map  poro, 
from Load_Petrophysical_data  import read_eclipse_keyword
#file_path = ("../../0_Volve_dataset/7_Reservoir_Model-Eclipse_model/Volve_sim_model_PPA-Eclipse Res Model/VOLVE_2016_EXPAND.DATA")

SCRIPT_DIR = Path(__file__).resolve().parent

file_path = (
    SCRIPT_DIR.parent.parent 
    / "0_Volve_dataset" 
    / "7_Reservoir_Model-Eclipse_model" 
    / "Volve_sim_model_PPA-Eclipse Res Model" 
    / "VOLVE_2016_EXPAND.DATA"
)

poro_values = read_eclipse_keyword(file_path, "PORO")
perm_values = read_eclipse_keyword(file_path, "PERMX")


# In[5]:


# check that the lenght of the poro and perm list is the same size at the length of the grid 
len(poro_values)==len(perm_values)==108 * 100 * 63


# In[6]:


# change the 1d list to a 3d list and then to a 2d list. for the 2d list average the z values
NX, NY, NZ = 108, 100, 63

poro_3d = poro_values.reshape((NZ, NY, NX))
poro_avg = poro_3d.mean(axis=0)

perm_3d = perm_values.reshape((NZ, NY, NX))
perm_avg = perm_3d.mean(axis=0)


# In[7]:


#Load the eclipse well location. This also include the wells from the production dataframe.
# you might have to go back to check that the location for Eclipse Wellname==NPD_WELL_BORE_NAME
#file_path = ("../../0_Volve_dataset/Processed_input/WELSPECS.xlsx")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(BASE_DIR, '0_Volve_dataset', 'Processed_input', 'WELSPECS.xlsx')

def load_data():

    return pd.read_excel(data_path)

WELSPECS_df = load_data()





# In[8]:


# change the I,J of the Poro grid to real x and y using MAPAXES
#MAPAXES
#432156.531 6476477.000
#432156.531 6481452.000
#437531.531 6481452.000

x_min = 432156.531
x_max = 437531.531

y_min = 6476477.000
y_max = 6481452.000

x_real = np.linspace(x_min, x_max, NX)
y_real = np.linspace(y_min, y_max, NY)


# ## Load the production data frame and do all calcs

# In[9]:


from Load_clean_production_data import Load_clean_production_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(BASE_DIR, '0_Volve_dataset', '5_Production_data', 'Volve_production_data.xlsx')

def load_data():

    return pd.read_excel(data_path)

production_df = load_data()

from Load_clean_production_data import metric_to_field
production_df_field=pd.DataFrame()
production_df_field=metric_to_field(production_df)


# In[10]:


# drop columns u dont need
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


# In[11]:


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


# In[12]:


# change the lat lon to xy
from pyproj import Transformer

transformer = Transformer.from_crs(
    "EPSG:4326",   # WGS84 lat/lon
    "EPSG:32631",  # UTM zone 31N
    always_xy=True
)
production_df_field["Easting_m"], production_df_field["Northing_m"] = transformer.transform(production_df_field["lon_dd"].values,production_df_field["lat_dd"].values)


# In[13]:


# caculate cummulative production
from Load_clean_production_data import cummulative_calc
production_df_field=cummulative_calc(production_df_field)

# calculate running time and normalize time
from Load_clean_production_data import times_calc
production_df_field=times_calc(production_df_field)

# calculate GOR and WCUT
from Load_clean_production_data import ratio_calc
production_df_field=ratio_calc(production_df_field)


# In[14]:


production_df_field.columns


# In[19]:


production_df_field


# In[20]:
app = dash.Dash(
    __name__,
    assets_folder="../assets",
    routes_pathname_prefix='/',
    requests_pathname_prefix='/comparison/'
)

app2 = app

poro_heatmap=go.Heatmap(x=x_real,y=y_real,z=poro_avg,colorscale="Viridis",colorbar={"title": "Porosity<br>frac"})
perm_heatmap=go.Heatmap(x=x_real,y=y_real,z=perm_avg,colorscale="Viridis",colorbar={"title": "Permeability<br>md"})
wellnames =production_df_field["NPD_WELL_BORE_NAME"].unique()
app.layout=html.Div([
                    # basic header
                     html.Div([html.H1('Well Comparison Dashboard')],style={"border": "1px solid black","height": "4vh","textAlign": "center"}),
                    # three column div
                    html.Div([
                   # 4 comparasion plots
                            html.Div([
                                    html.Div([
                                            dcc.Graph(id="plot1", style={"height": "90%", "marginLeft": "15px","marginRight": "10px"}),
                                            dcc.Dropdown(id="y_axis_dropdown",
                                                         className="rotate-dropdown",
                                                         options=[
                                                                {"label": "Oil Rate", "value": "BORE_OIL_VOL_STB"},
                                                                {"label": "Water Rate", "value": "BORE_WAT_VOL_STB"},
                                                                {"label": "Gas Rate", "value": "BORE_GAS_VOL_MSCF"},
                                                                {"label": "Cum. Oil Prod.", "value": "Cummulative_Oil_VOL_STB"},
                                                                {"label": "Cum. Gas Prod.", "value": "Cummulative_Gas_VOL_MSCF"},
                                                                {"label": "Cum. Water Prod.", "value": "Cummulative_WAT_VOL_STB"},
                                                                {"label": "GOR", "value": "GOR_MSCF/STB"},
                                                                {"label": "WCUT_%.", "value": "WCUT_%"},
                                                                {"label": "WHP", "value": "AVG_WHP_P_PSI"},
                                                                 ],
                                                                value="BORE_OIL_VOL_STB",
                                                                clearable=False,
                                                         style={"position": "absolute", "top": "350px","left": "-55px","width": "150px","zIndex": 1000,"transform": "translateY(-50%)"}
                                                        ),
                                            dcc.Dropdown(id="x_axis_dropdown",
                                                         options=[
                                                                {"label": "Date", "value": "DATEPRD"},
                                                                {"label": "Normalized Time", "value": "Normalized_Time_days"},
                                                                {"label": "Days on Prod.", "value": "Days_on_prod"},
                                                                 ],
                                                                value="DATEPRD",
                                                                clearable=False,
                                                         style={"position": "absolute",  "transform": "translateX(-50%)","width": "150px",  "zIndex": 1000,"left": "50%","bottom": "7%" }
                                                        )
                                        
                                            ], style={"border": "1px solid gray","width": "50%","height": "50%","boxSizing": "border-box","position": "relative"}),
                                    # plot 2
                                    html.Div([
                                            dcc.Graph(id="plot2", style={"height": "90%", "marginLeft": "15px","marginRight": "10px"}),
                                            dcc.Dropdown(id="y_axis_dropdown_2",
                                                         className="rotate-dropdown",
                                                         options=[
                                                                {"label": "Oil Rate", "value": "BORE_OIL_VOL_STB"},
                                                                {"label": "Water Rate", "value": "BORE_WAT_VOL_STB"},
                                                                {"label": "Gas Rate", "value": "BORE_GAS_VOL_MSCF"},
                                                                {"label": "Cum. Oil Prod.", "value": "Cummulative_Oil_VOL_STB"},
                                                                {"label": "Cum. Gas Prod.", "value": "Cummulative_Gas_VOL_MSCF"},
                                                                {"label": "Cum. Water Prod.", "value": "Cummulative_WAT_VOL_STB"},
                                                                {"label": "GOR", "value": "GOR_MSCF/STB"},
                                                                {"label": "WCUT_%.", "value": "WCUT_%"},
                                                                {"label": "WHP", "value": "AVG_WHP_P_PSI"},
                                                                 ],
                                                                value="BORE_OIL_VOL_STB",
                                                                clearable=False,
                                                         style={"position": "absolute","top": "350px","left": "-55px","width": "150px","zIndex": 1000,"transform": "translateY(-50%)"}
                                                        ),
                                            dcc.Dropdown(id="x_axis_dropdown_2",
                                                         options=[
                                                                {"label": "Date", "value": "DATEPRD"},
                                                                {"label": "Normalized Time", "value": "Normalized_Time_days"},
                                                                {"label": "Days on Prod.", "value": "Days_on_prod"},
                                                                 ],
                                                                value="DATEPRD",
                                                                clearable=False,
                                                         style={"position": "absolute",  "transform": "translateX(-50%)","width": "150px",  "zIndex": 1000,"left": "50%","bottom": "7%" }
                                                        )
                                            ], style={"border": "1px solid gray","width": "50%","height": "50%","boxSizing": "border-box","position": "relative"}),
                                    #plot3
                                    html.Div([
                                            dcc.Graph(id="plot3", style={"height": "90%", "marginLeft": "15px","marginRight": "10px"}),
                                            dcc.Dropdown(id="y_axis_dropdown_3",
                                                         className="rotate-dropdown",
                                                         options=[
                                                                {"label": "Oil Rate", "value": "BORE_OIL_VOL_STB"},
                                                                {"label": "Water Rate", "value": "BORE_WAT_VOL_STB"},
                                                                {"label": "Gas Rate", "value": "BORE_GAS_VOL_MSCF"},
                                                                {"label": "Cum. Oil Prod.", "value": "Cummulative_Oil_VOL_STB"},
                                                                {"label": "Cum. Gas Prod.", "value": "Cummulative_Gas_VOL_MSCF"},
                                                                {"label": "Cum. Water Prod.", "value": "Cummulative_WAT_VOL_STB"},
                                                                {"label": "GOR", "value": "GOR_MSCF/STB"},
                                                                {"label": "WCUT_%.", "value": "WCUT_%"},
                                                                {"label": "WHP", "value": "AVG_WHP_P_PSI"},
                                                                 ],
                                                                value="BORE_OIL_VOL_STB",
                                                                clearable=False,
                                                         style={"position": "absolute", "top": "350px","left": "-55px","width": "150px","zIndex": 1000,"transform": "translateY(-50%)"}
                                                        ),
                                            dcc.Dropdown(id="x_axis_dropdown_3",
                                                         options=[
                                                                {"label": "Date", "value": "DATEPRD"},
                                                                {"label": "Normalized Time", "value": "Normalized_Time_days"},
                                                                {"label": "Days on Prod.", "value": "Days_on_prod"},
                                                                 ],
                                                                value="DATEPRD",
                                                                clearable=False,
                                                         style={"position": "absolute",  "transform": "translateX(-50%)","width": "150px",  "zIndex": 1000,"left": "50%","bottom": "7%" }
                                                        )
                                            ], style={"border": "1px solid gray","width": "50%","height": "50%","boxSizing": "border-box","position": "relative"}),

                                    #plot4
                                    html.Div([
                                            dcc.Graph(id="plot4", style={"height": "90%", "marginLeft": "15px","marginRight": "10px"}),
                                            dcc.Dropdown(id="y_axis_dropdown_4",
                                                         className="rotate-dropdown",
                                                         options=[
                                                                {"label": "Oil Rate", "value": "BORE_OIL_VOL_STB"},
                                                                {"label": "Water Rate", "value": "BORE_WAT_VOL_STB"},
                                                                {"label": "Gas Rate", "value": "BORE_GAS_VOL_MSCF"},
                                                                {"label": "Cum. Oil Prod.", "value": "Cummulative_Oil_VOL_STB"},
                                                                {"label": "Cum. Gas Prod.", "value": "Cummulative_Gas_VOL_MSCF"},
                                                                {"label": "Cum. Water Prod.", "value": "Cummulative_WAT_VOL_STB"},
                                                                {"label": "GOR", "value": "GOR_MSCF/STB"},
                                                                {"label": "WCUT_%.", "value": "WCUT_%"},
                                                                {"label": "WHP", "value": "AVG_WHP_P_PSI"},
                                                                 ],
                                                                value="BORE_OIL_VOL_STB",
                                                                clearable=False,
                                                         style={"position": "absolute","top": "350px","left": "-55px","width": "150px","zIndex": 1000,"transform": "translateY(-50%)"}
                                                        ),
                                            dcc.Dropdown(id="x_axis_dropdown_4",
                                                         options=[
                                                                {"label": "Date", "value": "DATEPRD"},
                                                                {"label": "Normalized Time", "value": "Normalized_Time_days"},
                                                                {"label": "Days on Prod.", "value": "Days_on_prod"},
                                                                 ],
                                                                value="DATEPRD",
                                                                clearable=False,
                                                         style={"position": "absolute",  "transform": "translateX(-50%)","width": "150px",  "zIndex": 1000,"left": "50%","bottom": "7%" }
                                                        )
                                            ], style={"border": "1px solid gray","width": "50%","height": "50%","boxSizing": "border-box","position": "relative"}),
        
                                    ], style={"display": "flex","flexWrap": "wrap", "width": "70vw","height": "95vh"}),
        
                             # permeability and porosity heat map
                             html.Div([
                                     # poro
                                     html.Div([
                                              dcc.Graph(
                                                        id="porosity",
                                                        figure={"data":poro_heatmap,                                                      
                                                              "layout":go.Layout(title={"text": "Porosity Heatmap", "font": {"size": 40}}, 
                                                                                 hovermode ="closest", 
                                                                                 autosize=True
                                                                                )
                                                               },
                                                       style={"width": "100%", "height": "100%"}  # this style is for the graph itself
                                                       )
                                                ],style={"border": "1px solid gray", "width": "100%", "height": "50%","overflow": "hidden" }), # this style is for layout of the div
                    
                                     # permeability
                                     html.Div([
                                              dcc.Graph(id="perm",
                                                        figure={"data":perm_heatmap,                                                      
                                                              "layout":go.Layout(title={"text": "Permeability Heatmap", "font": {"size": 40}}, 
                                                                                 hovermode ="closest",
                                                                                 autosize=True
                                                                                )
                                                               },
                                                        style={"width": "100%", "height": "100%"} # this style is for the graph itself
                                                       )
                                             ],style={"border": "1px solid gray", "width": "100%", "height": "50.1%","overflow": "hidden"}),  
                                 
                                     ],style={"border": "1px solid gray", "display": "flex","width": "24vw","height": "95vh", "flexDirection": "column"}), # this style is for the perm and poro heatmap together        
                                 
                           # Well Dropdown
                             html.Div([
                                     html.H3("Select Wells"),
                                     dcc.Checklist(id="well_checklist",options=wellnames,value=[]), # Make sure this list correspond to the well names you want.production_df_field  has eclipse wellnames and production wellnames but eclipse location. 
                                     ], style={"border": "1px solid gray","width": "5vw","padding": "10px"})
                             
                             ], style={"display": "flex","width": "100vw","height": "95vh"})
        
    
    
    

    
                    ],style={"border": "1px solid gray", "width": "100vw", "height": "100vh"})

#--------------------------------------------------------------------------------------------------------------------
# update heatmap with wells location
#---------------------------------------------------------------------------------------------------------------------
@app.callback(Output("porosity", "figure"),
              [Input("well_checklist", "value")])
def update_poro_heatmap(selected_wells):
    fig = go.Figure()
    fig.add_trace(poro_heatmap)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]

    fig.add_trace(go.Scatter(x=filtered_df["Easting_m"],y=filtered_df["Northing_m"],mode="markers+text",
                             text=filtered_df["NPD_WELL_BORE_NAME"],textposition="top center",marker={"size": 10, "color": "blue"},name="Selected Wells"))
    fig.update_layout(title={"text": "Porosity Heatmap", "font": {"size": 40}}, hovermode ="closest", autosize=True,plot_bgcolor="white",paper_bgcolor="white",
                      xaxis={"showgrid": False,"zeroline": False},yaxis={"showgrid": False,"zeroline": False,"scaleanchor": "x"})

    return fig

@app.callback(Output("perm", "figure"),
              [Input("well_checklist", "value")])
def update_perm_heatmap(selected_wells):
    fig = go.Figure()
    fig.add_trace(perm_heatmap)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]

    fig.add_trace(go.Scatter(x=filtered_df["Easting_m"],y=filtered_df["Northing_m"],mode="markers+text",
                             text=filtered_df["NPD_WELL_BORE_NAME"],textposition="top center",marker={"size": 10, "color": "blue"},name="Selected Wells"))
    fig.update_layout(title={"text": "Permeability Heatmap", "font": {"size": 40}}, hovermode ="closest", autosize=True,plot_bgcolor="white",paper_bgcolor="white",
                      xaxis={"showgrid": False,"zeroline": False},yaxis={"showgrid": False,"zeroline": False,"scaleanchor": "x"})

    return fig


#--------------------------------------------------------------------------------------------------------------------
# update the production plots
#---------------------------------------------------------------------------------------------------------------------
# plot1
@app.callback(Output("plot1", "figure"),
              [Input("well_checklist", "value"),
               Input("y_axis_dropdown", "value"),
              Input("x_axis_dropdown", "value")])
def plot1(selected_wells,y_axis,x_axis):
    fig = go.Figure()
    #print(selected_wells)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]
    for well in selected_wells:
        df_well = filtered_df[filtered_df["NPD_WELL_BORE_NAME"] == well]


        fig.add_trace(go.Scatter(x=df_well[x_axis],y=df_well[y_axis],mode="lines", text=df_well["NPD_WELL_BORE_NAME"],name=well))
        fig.update_layout(title={"text": f"{x_axis} vs {y_axis}", "font": {"size": 22}},xaxis_title=x_axis, yaxis_title=y_axis,hovermode="closest")

    return fig

# plot2
@app.callback(Output("plot2", "figure"),
              [Input("well_checklist", "value"),
               Input("y_axis_dropdown_2", "value"),
              Input("x_axis_dropdown_2", "value")])
def plot2(selected_wells,y_axis,x_axis):
    fig = go.Figure()
    #print(selected_wells)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]
    for well in selected_wells:
        df_well = filtered_df[filtered_df["NPD_WELL_BORE_NAME"] == well]


        fig.add_trace(go.Scatter(x=df_well[x_axis],y=df_well[y_axis],mode="lines", text=df_well["NPD_WELL_BORE_NAME"],name=well))
        fig.update_layout(title={"text": f"{x_axis} vs {y_axis}", "font": {"size": 22}},xaxis_title=x_axis, yaxis_title=y_axis,hovermode="closest")

    return fig

# plot3
@app.callback(Output("plot3", "figure"),
              [Input("well_checklist", "value"),
               Input("y_axis_dropdown_3", "value"),
              Input("x_axis_dropdown_3", "value")])
def plot3(selected_wells,y_axis,x_axis):
    fig = go.Figure()
    #print(selected_wells)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]
    for well in selected_wells:
        df_well = filtered_df[filtered_df["NPD_WELL_BORE_NAME"] == well]


        fig.add_trace(go.Scatter(x=df_well[x_axis],y=df_well[y_axis],mode="lines", text=df_well["NPD_WELL_BORE_NAME"],name=well))
        fig.update_layout(title={"text": f"{x_axis} vs {y_axis}", "font": {"size": 22}},xaxis_title=x_axis, yaxis_title=y_axis,hovermode="closest")

    return fig

#plot4
@app.callback(Output("plot4", "figure"),
              [Input("well_checklist", "value"),
               Input("y_axis_dropdown_4", "value"),
              Input("x_axis_dropdown_4", "value")])
def plot4(selected_wells,y_axis,x_axis):
    fig = go.Figure()
    #print(selected_wells)

    # only add wells if selected
    filtered_df = production_df_field[production_df_field["NPD_WELL_BORE_NAME"].isin(selected_wells)]
    for well in selected_wells:
        df_well = filtered_df[filtered_df["NPD_WELL_BORE_NAME"] == well]


        fig.add_trace(go.Scatter(x=df_well[x_axis],y=df_well[y_axis],mode="lines", text=df_well["NPD_WELL_BORE_NAME"],name=well))
        fig.update_layout(title={"text": f"{x_axis} vs {y_axis}", "font": {"size": 22}},xaxis_title=x_axis, yaxis_title=y_axis,hovermode="closest")

    return fig

if __name__ == '__main__':
    app.run()


# In[ ]:




