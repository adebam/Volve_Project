# wsgi.py
import dash
import sys
import os
from dash import html, dcc
from dash.dependencies import Input, Output
from flask import Flask, render_template_string
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Force Python to see inside your visualization source folder
sys.path.append(os.path.join(os.path.dirname(__file__), '1_Visualization', 'src'))

# Clean, safe imports
from Dash_Production__Render import app as app1
from Dash_Well_Comparison_Render import app as app2
from Dash_Decline_Curve_Render import app as app3

# 1. Create a central Flask server to act as the main traffic controller
server = Flask(__name__)

# 2. Design a clean, simple landing page layout for your root URL (/)
@server.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Volve Project Dashboards</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f7f9fa; color: #333; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #1a2b4c; border-bottom: 2px solid #e1e4e8; padding-bottom: 10px; }
                ul { list-style: none; padding: 0; }
                li { margin: 15px 0; }
                a { display: block; padding: 12px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center; transition: background 0.2s; }
                a:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Volve Project Analytics</h1>
                <p>Select a dashboard module to view the production data:</p>
                <ul>
                    <li><a href="/production/">Production Dashboard</a></li>
                    <li><a href="/comparison/">Well Comparison Dashboard</a></li>
                    <li><a href="/decline/">Decline Curve Analysis</a></li>
                </ul>
            </div>
        </body>
        </html>
    ''')

# 3. CRITICAL STEP: Configure Dash requests paths to respect the sub-directories
app1.config.update({
    'routes_pathname_prefix': '/production/',
    'requests_pathname_prefix': '/production/'
})

app2.config.update({
    'routes_pathname_prefix': '/comparison/',
    'requests_pathname_prefix': '/comparison/'
})

app3.config.update({
    'routes_pathname_prefix': '/decline/',
    'requests_pathname_prefix': '/decline/'
})



# 4. Bind the 3 isolated dash apps to their respective Flask sub-paths
server.wsgi_app = DispatcherMiddleware(server.wsgi_app, {
    '/production': app1.server,
    '/comparison': app2.server,
    '/decline': app3.server
})
if __name__ == '__main__':
    # This is only used for testing locally
    from werkzeug.serving import run_simple
    run_simple('localhost', 8050, server, use_reloader=True, use_debugger=True)
