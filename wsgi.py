# wsgi.py
import dash
import sys
import os
from dash import html, dcc
from dash.dependencies import Input, Output

# Import the dash app objects from your scripts
# (Ensure your scripts expose the 'app' or 'server' variable)
from find_visualization.src.Dash_Production__Render import app as app1
from find_visualization.src.Dash_Well_Comparison_Render import app as app2
from find_visualization.src.Dash_Decline_Curve_Render import app as app3

# Create a master app to handle navigation
meta_app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = meta_app.server  # Render will look for this 'server' target

meta_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link('Production Dashboard | ', href='/production'),
        dcc.Link('Well Comparison | ', href='/comparison'),
        dcc.Link('Decline Curve Analysis', href='/decline')
    ], style={'padding': '10px', 'background': '#f4f4f4'}),
    html.Div(id='page-content')
])

@meta_app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/production':
        return app1.layout
    elif pathname == '/comparison':
        return app2.layout
    elif pathname == '/decline':
        return app3.layout
    else:
        return html.H3("Welcome! Please select a dashboard from the navigation bar above.")

if __name__ == '__main__':
    meta_app.run_server(debug=False)
