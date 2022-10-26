from dash import dcc, html
import numpy as np
import requests
import os



def create_pred_layout(api_url='http://127.0.0.1:5000') :
    main_features = ['AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']
    url = api_url + '/api/ids'
    response = requests.get(url)
    ids_list = response.json()['ids']
    short_id_list = list(np.random.choice(ids_list, size=50, replace=False)) 


    prediction_tab = dcc.Tab(label='Prédiction', children = [
                #html.Button('test-button',id="test_button", style={'width' : '100%', 'height' : '60px', 'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H3(children='Choix du Modèle :', style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        html.Div(dcc.Dropdown(id="model_dropdown2", multi=False), style={'width' : '69%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H3(children='Identifiant :', style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        dcc.Dropdown(short_id_list, value=short_id_list[0],id="id_value", style={'width' : '69%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H3(children='Méthode d\'imputation :', style={'width' : '50%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        dcc.Dropdown(['None','median','mean'],value='None',id="Imputer_method", multi=False, style={'width' : '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        #html.Button('Imputer',id="reset", style={'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'width' : '100%', 'display': 'inline-block', 'vertical-align': 'top'}),            
                html.Div([
                    html.H2('Variables Principales', style={"border":"1px black solid", 'textAlign': 'center', 'vertical-align': 'top'}),
                    html.Div(id='main_features', children=[
                        html.Div([
                        html.H4(feat),
                        dcc.Input(id='input_%s' %(feat), type="number", value=None, debounce = True, style={'width' : '75%'})
                    ], style={'display': 'inline-block', 'vertical-align': 'top', }) for feat in main_features
                ])], style={'height' : '140px' }),
                #html.Button('Est. taux',id="est_calcul"),
                html.Div([
                    html.Div([
                        html.H4('Taux d\'intérêt :',style={'width' : '50%', 'display': 'inline-block'}),
                        #html.Div(id='est_loan_rate',style={'width' : '20%', 'display': 'inline-block'})], 
                        dcc.Input(id='est_loan_rate',type = 'number', value =0, min=0, max=1, step=0.01, style={'width' : '20%', 'display': 'inline-block'})], 
                        style={'width' : '15%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H4('Durée remboursement :',style={'width' : '50%', 'display': 'inline-block'}),
                        html.Div(id='est_time',style={'width' : '20%', 'display': 'inline-block'})], 
                        style={'width' : '50%', 'display': 'inline-block', 'vertical-align': 'top'})
                ], style={'height' : '40px' }),
                html.Div([
                    html.Button(html.H3('Variables Secondaires'), id='hide_sec_features_button', style={"border":"1px black solid",'width' : '100%', 'textAlign': 'center', 'vertical-align': 'top', 'backgroundColor': 'white'}),
                    html.Div([
                        html.Div(html.Div(dcc.Tabs([],id='tabs_features'), id="model_features_box"), style={'width' : '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                        html.Div(dcc.Tabs([
                            dcc.Tab(children = dcc.Graph(id='polar_graph_feat',style={'height' : '350px'}), label='Graphe Polaire'),
                            dcc.Tab(children = dcc.Graph(id='boxplot_graph_feat',style={'height' : '350px'}), label='Boxplot')
                        ]), style={'width' : '39%', 'display': 'inline-block', 'vertical-align': 'top'})
                    ]),
                ], style={'height' : '500px'}),
                html.Button(html.H3('Predire'),id="predict", style={'width' : '100%', 'height' : '60px', 'textAlign': 'center'}),
                html.Div([
                    html.Div([
                            html.H4('Probabilité de remboursement :',style={'width' : '70%', 'display': 'inline-block'}),
                            html.Div(id='prediction_value',style={'width' : '29%', 'display': 'inline-block'})], 
                            style={'width' : '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                            html.H4('Status conseillé par le modèle :',style={'width' : '70%', 'display': 'inline-block'}),
                            html.Div(id='prediction_status')], 
                            style={'width' : '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'width' : '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.Div([
                            html.H4('Bénéf. max :',style={'width' : '70%', 'display': 'inline-block'}),
                            html.Div(id='max_gain',style={'width' : '29%', 'display': 'inline-block'})], 
                            style={'width' : '25%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                            html.H4('Bénéf. espérés par le modèle :',style={'width' : '70%', 'display': 'inline-block'}),
                            html.Div(id='prediction_gain',style={'width' : '29%', 'display': 'inline-block'})], 
                            style={'width' : '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                            html.H4('Ratio bénéf. espéré/max :',style={'width' : '70%', 'display': 'inline-block'}),
                            html.Div(id='ratio_gain',style={'width' : '29%', 'display': 'inline-block'})], 
                            style={'width' : '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'width' : '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                #dcc.Store(id='contribs'),
                html.Div([html.H4('Influence des variables sur la prédiction', style={'textAlign': 'center'}),
                html.Div(dcc.Graph(id='waterfall_figure'))
                ])
            ])
    return prediction_tab