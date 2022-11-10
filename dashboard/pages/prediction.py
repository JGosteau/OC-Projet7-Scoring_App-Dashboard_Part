from dash import dcc, html
import numpy as np
import requests
import os



def create_pred_layout_pred(short_id_list) :


    prediction_tab = dcc.Tab(label='Prédiction', children = [
                html.Div([
                    html.Div([
                        html.H3(children='Choix du Modèle :', style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        html.Div(dcc.Dropdown(id="model_dropdown3", multi=False), style={'width' : '69%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H3(children='Identifiant :', style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        dcc.Dropdown(short_id_list, value=short_id_list[0],id="id_value2", style={'width' : '69%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.H3(children='Méthode d\'imputation :', style={'width' : '50%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        dcc.Dropdown(['None','median','mean'],value='None',id="Imputer_method2", multi=False, style={'width' : '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    ], style={'width' : '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'width' : '100%', 'display': 'inline-block', 'vertical-align': 'top'}),          
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
                
                html.Div([html.H4('Status : ',style={'display': 'inline-block'}), html.Div(id='pred_calc_status',style={'display': 'inline-block'})]),
                html.Div([html.H4('Influence des variables sur la prédiction', style={'textAlign': 'center'}),
                html.Div(dcc.Graph(id='waterfall_figure'))
                ])
            ])
    return prediction_tab