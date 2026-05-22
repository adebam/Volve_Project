# wsgi.py
import sys
import os

# Tell Python to look inside the 1_Visualization/src folder for your scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '1_Visualization', 'src')))

# Import the server instance from your Dash script
from Dash_Production__Render import app

# Expose the Flask server for Gunicorn
server = app.server
