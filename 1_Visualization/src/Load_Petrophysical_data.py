"""
This loads data for the ecplise data file using a keyword.
"""

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

#others
from pathlib import Path
import re

def read_eclipse_keyword(file_path, keyword):

    text = Path(file_path).read_text(
        encoding="latin-1",
        errors="ignore"
    )

    # remove comments
    text = re.sub(r"--.*", "", text)

    # capture keyword block
    pattern = rf"\b{keyword}\b\s*(.*?)\s*/"

    match = re.search(
        pattern,
        text,
        flags=re.S | re.I
    )

    if match is None:
        raise ValueError(f"{keyword} not found")

    block = match.group(1)

    values = []

    for token in block.split():

        if "*" in token:
            n, val = token.split("*")
            values.extend([float(val)] * int(n))

        else:
            values.append(float(token))

    return np.array(values, dtype=float)

