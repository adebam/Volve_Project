"""
def Load_clean_production_data(file_path)-- loads the production data
def metric_to_field(df)-- the production is in metric. This changes to field. returns a new dataframe production_df_field
def cummulative_calc(df)-- calculate cumulative production.  watch the channels. Only works with the specific channel name.
ratio_calc(df)-- calculate ratios. Again watch the channels
def times_calc(df)-- caculates 
1. Normalized_Time_days. Changes Date to running day. Makes all time for all well start at day 0
2. is_producing. This is Normalized_Time_days but not counting the time the well is not producing
"""
#basics
import pandas as pd
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


def Load_clean_production_data(file_path):
    #file_path = ("../../0_Volve_dataset/5_Production_data/Volve_production_data.xlsx")
    production_df=pd.read_excel(file_path)
    return production_df

def metric_to_field(df):
    production_df_field=pd.DataFrame()
    production_df_field["DATEPRD"] = df["DATEPRD"] 
    production_df_field["WELL_BORE_CODE"] = df["WELL_BORE_CODE"]
    production_df_field["NPD_WELL_BORE_CODE"] = df["NPD_WELL_BORE_CODE"] 
    production_df_field["NPD_WELL_BORE_NAME"] = df["NPD_WELL_BORE_NAME"] 
    production_df_field["NPD_FIELD_CODE"] = df["NPD_FIELD_CODE"] 
    production_df_field["NPD_FIELD_NAME"] = df["NPD_FIELD_NAME"] 
    production_df_field["NPD_FACILITY_CODE"] = df["NPD_FACILITY_CODE"] 
    production_df_field["NPD_FACILITY_NAME"] = df["NPD_FACILITY_NAME"] 
    production_df_field["ON_STREAM_HRS"] = df["ON_STREAM_HRS"] 
    production_df_field["AVG_DOWNHOLE_PRESSURE_PSI"] = df["AVG_DOWNHOLE_PRESSURE"] * 14.5038
    production_df_field["AVG_DOWNHOLE_TEMPERATURE_F"] = df["AVG_DOWNHOLE_TEMPERATURE"] * 9/5 + 32
    production_df_field["AVG_DP_TUBING_PSI"] = df["AVG_DP_TUBING"] * 14.5038
    production_df_field["AVG_ANNULUS_PRESS_PSI"] = df["AVG_ANNULUS_PRESS"] * 14.5038
    production_df_field["AVG_CHOKE_SIZE_P"] = df["AVG_CHOKE_SIZE_P"] 
    production_df_field["AVG_CHOKE_UOM"] = df["AVG_CHOKE_UOM"] 
    production_df_field["AVG_WHP_P_PSI"] = df["AVG_WHP_P"]* 14.5038 
    production_df_field["AVG_WHT_P"] = df["AVG_WHT_P"]* 9/5 + 32 
    production_df_field["DP_CHOKE_SIZE"] = df["DP_CHOKE_SIZE"] 
    production_df_field["BORE_OIL_VOL_STB"] = df["BORE_OIL_VOL"]* 6.28981
    production_df_field["BORE_GAS_VOL_MSCF"] = df["BORE_GAS_VOL"] * 35.3147/1000 
    production_df_field["BORE_WAT_VOL_STB"] = df["BORE_WAT_VOL"]* 6.28981
    production_df_field["BORE_WI_VOL"] = df["BORE_WI_VOL"]
    production_df_field["FLOW_KIND"] = df["FLOW_KIND"]
    production_df_field["WELL_TYPE"] = df["WELL_TYPE"]

    return production_df_field

def cummulative_calc(df):
    # calculate cummulative production
    df["Cummulative_Oil_VOL_STB"] = df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).groupby("NPD_WELL_BORE_NAME")["BORE_OIL_VOL_STB"].cumsum()
    df["Cummulative_Gas_VOL_MSCF"] = df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).groupby("NPD_WELL_BORE_NAME")["BORE_GAS_VOL_MSCF"].cumsum()
    df["Cummulative_WAT_VOL_STB"] = df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).groupby("NPD_WELL_BORE_NAME")["BORE_WAT_VOL_STB"].cumsum()
    return df

def ratio_calc(df):
    df["GOR_MSCF/STB"]=df["BORE_GAS_VOL_MSCF"]/df["BORE_OIL_VOL_STB"] # gas oil ratio
    df["WCUT_%"]=(df["BORE_WAT_VOL_STB"]/(df["BORE_WAT_VOL_STB"]+df["BORE_OIL_VOL_STB"])) *100 # water cut
    return df
        

def times_calc(df):
    # calculate normalize time and running day
    df["Time_zero"]=(df.groupby("NPD_WELL_BORE_NAME")["DATEPRD"].transform("min"))
    #Normalized time to time zero
    df["Normalized_Time_days"]=(df["DATEPRD"]-df["Time_zero"]).dt.days.astype(float)
    #Days on production no downtime
    df["is_producing"] = (
        (df["BORE_OIL_VOL_STB"] > 0) |
        (df["BORE_GAS_VOL_MSCF"] > 0) |
        (df["BORE_WAT_VOL_STB"] > 0))
    
    # Days_on_prod: count only producing days, grouped by well
    df["Days_on_prod"] = (df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).groupby("NPD_WELL_BORE_NAME")["is_producing"].cumsum())-0
    return df
