import pandas as pd
from dash import Dash, dcc, html
import requests
import numpy as np
from dashboard.pages.info_indiv import create_pred_layout_indiv
from dashboard.pages.prediction import create_pred_layout_pred
from dashboard.pages.model import model_tab
from dashboard.pages.callback import register_callbacks

api_url = 'http://127.0.0.1:5000'
api_url = 'http://jgosteau1.pythonanywhere.com'

url = api_url + '/api/ids'
response = requests.get(url)
ids_list = response.json()['ids']
short_id_list = list(np.random.choice(ids_list, size=50, replace=False))

app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='indiv_tabs'),
    dcc.Store(id='indiv_data'),
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
        create_pred_layout_indiv(short_id_list),
        create_pred_layout_pred(short_id_list),
        model_tab
    ]),

])

register_callbacks(app, api_url)

if __name__ == '__main__' :
    app.run_server(debug = True)