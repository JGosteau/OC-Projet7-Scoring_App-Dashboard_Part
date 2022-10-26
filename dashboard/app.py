import pandas as pd
from dash import Dash, dcc, html

from dashboard.pages.prediction import create_pred_layout
from dashboard.pages.model import model_tab
from dashboard.pages.callback import register_callbacks

api_url = 'http://127.0.0.1:5000'
#api_url = 'http://albertkrif.pythonanywhere.com'

app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='datainfo'),
    dcc.Store(id='model_list'),
    dcc.Store(id='model_features'),
    dcc.Store(id='model_trigger_var'),
    html.H2(
        children='Application de scoring',
        style={
            'textAlign': 'center',
        }
    ),
    dcc.Tabs([
        create_pred_layout(api_url),
        model_tab
    ]),

])

register_callbacks(app, api_url)

if __name__ == '__main__' :
    app.run_server(debug = True)