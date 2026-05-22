"""
def Fetch_XY_location_make_dataframe(root_dir) -- makes a dataframe  of xy location. Needs clean_value(value) & get_file_type(file_path)
def dms_to_dd(dms_str) -- changes degrees, minutes, seconds to decimal degrees
def DMS_decimal(df) -- applies def dms_to_dd(dms_str) to a dataframe
def poly(file_path)-- only fetches 2014_Volve_Hugin_Base.dat polygon and turn it into a dataframe. it can be later be plotted. 

pyproj is needed to be installed
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

#others
from pathlib import Path
import re


def clean_value(value):
    value = value.strip()
    value = re.sub(r"(?<=\d)\s*m$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\d)\s*deg\.?$", "", value, flags=re.IGNORECASE)
    return value.strip()

def get_file_type(file_path):
    name = file_path.name.upper()

    if "ACTUAL" in name:
        return "ACTUAL"
    elif "PLAN" in name:
        return "PLAN"
    else:
        return None

def Fetch_XY_location_make_dataframe(root_dir):
    """
    The codes below goes to the well technical data folder.
    Looks at all the files that has PLAN or ACTUAL in the file name.
    Extracts all data in the file with "WELL NAME", "WELLBORE NAME", "Surface EW", "Surface NS","Surface Latitude","Surface Longitude", "Bottom Hole EW", "Bottom Hole NS"
    Counts all letters in all the wells names. I don't want wellnames with too many letters.
    Makes a dataframe of all the information.
    """
    #root_dir = Path("../../Projects/0_Volve_dataset/13_Well_technical_data")
    fields = [
        "WELL NAME",
        "WELLBORE NAME",
        "Surface EW",
        "Surface NS",
        "Surface Latitude",
        "Surface Longitude",
        "Bottom Hole EW",
        "Bottom Hole NS",
    ]
    patterns = {
        field: re.compile(rf"^{re.escape(field)}:\s*(.+?)\s*$", re.IGNORECASE)
        for field in fields
    }
    
    rows = []
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
    
        file_type = get_file_type(file_path)
    
        # only keep files that have ACTUAL or PLAN in the file name
        if file_type is None:
            continue
    
        row = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "plan_or_actual": file_type,
        }
    
        for field in fields:
            row[field] = None
    
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for field, pattern in patterns.items():
                        match = pattern.match(line)
                        if match:
                            row[field] = clean_value(match.group(1))
    
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue
    
        rows.append(row)
    
    df_headers = pd.DataFrame(rows)
    
    numeric_cols = [
        "Surface EW",
        "Surface NS",
        "Bottom Hole EW",
        "Bottom Hole NS",
    ]
    
    for col in numeric_cols:
        df_headers[col] = pd.to_numeric(df_headers[col], errors="coerce")
    
    df_headers=df_headers[df_headers['Surface EW'].notna()] # make sure the surface location is at least a number
    df_headers=df_headers.drop_duplicates(subset=['WELLBORE NAME']) # drop duplicate wellbore names
    df_headers["char_count"]=df_headers["WELLBORE NAME"].str.len()  # count and drop rows that have wellbore name more than 15
    df_headers= df_headers[df_headers["char_count"] <= 15]
    return df_headers



def dms_to_dd(dms_str):
    """
    changes the 'Surface Latitude','Surface Longitude' in degrees to  decimal and gives a new column lat_dd and lon_dd
    """
    # extract numbers
    parts = re.findall(r"[\d.]+", dms_str)
    degrees, minutes, seconds = map(float, parts)

    # extract direction (N, S, E, W)
    direction = re.search(r"[NSEW]", dms_str).group()

    dd = degrees + minutes/60 + seconds/3600

    if direction in ["S", "W"]:
        dd *= -1

    return dd
    

def DMS_decimal(df):
    """
    This applies def dms_to_dd(dms_str) to a df  to give new cols lat_dd and lon_dd
    """
    df["lat_dd"] = df["Surface Latitude"].apply(dms_to_dd)
    df["lon_dd"] = df["Surface Longitude"].apply(dms_to_dd)
    return df



def poly(file_path):
    """
    This fetches the 2014_Volve_Hugin_Base.dat from the dataset folder and makes it intoa dataframe.
    It load and convert the Hugin base polygon to a dataframe
    """
    
    #file_path = Path("../../0_Volve_dataset/2_Geophysical_Interpretations/Fault_polygons/2014_Volve_Hugin_Base.dat")
    
    rows = []
    current_polygon = None
    expected_points = None
    reading_points = False
    
    with open(file_path, "r", errors="ignore") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
    
        if line.startswith("Mapping Polygon"):
            parts = line.split()
            current_polygon = parts[-1]   # polygon sequence number
            reading_points = False
    
            # next line contains number of points
            i += 1
            info_line = lines[i].split()
            expected_points = int(info_line[2])
    
            reading_points = True
            point_count = 0
    
        elif reading_points and line.strip():
            parts = line.split()
    
            if len(parts) == 3:
                x, y, z = map(float, parts)
    
                rows.append({
                    "polygon_id": current_polygon,
                    "point_order": point_count + 1,
                    "x": x,
                    "y": y,
                    "z": z
                })
    
                point_count += 1
    
                if point_count >= expected_points:
                    reading_points = False
    
        i += 1
    
    df_poly = pd.DataFrame(rows)
    
    return df_poly


def transform_lat_long_to_XY(df_poly):
    """
    # covert polgyon coordinate to world map
    """

    from pyproj import Transformer
    
    # ED50 / UTM Zone 31N -> WGS84 lat/lon
    transformer = Transformer.from_crs(
        "EPSG:23031",
        "EPSG:4326",
        always_xy=True
    )
    
    df_poly["lon"], df_poly["lat"] = transformer.transform(
        df_poly["x"].values,
        df_poly["y"].values
    )
    
    return df_poly
