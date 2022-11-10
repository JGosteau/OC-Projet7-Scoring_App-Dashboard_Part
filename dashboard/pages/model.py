from dash import dcc, html


model_tab = dcc.Tab(label='Modèle', children = [
            html.Div([            
                html.Div([
                    html.H3(children='Choix du Modèle :', style={'width' : '10%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    html.Div(dcc.Dropdown(id="model_dropdown", multi=False), style={'width' : '89%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                ]),
                html.H4('Description du modèle :'),
                html.Div(id='model_desc', style={'width': '50%', 'height' : '120px',"border":"1px black solid"}),
                html.Div([
                    html.H4('Score ROC', style={'width': '80%', 'display': 'inline-block'}), 
                    html.Div(id='model_roc', style={'width': '19%', 'display': 'inline-block'}
                    )], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}                    ),
                html.Div([
                    html.Div([
                        html.H4('Taux d\'intérêt :', style={'width': '80%', 'display': 'inline-block'}),
                        html.Div(id="est_loan_rate2", style={'width': '19%', 'display': 'inline-block'}),
                    ], style={'width': '33%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4('Seuil conseillé : ', style={'display': 'inline-block', 'height' : '20px', 'width' : '80%'}),
                        html.Div(id='model_trigger_opt', style={'display': 'inline-block', 'height' : '20px', 'width' : '19%' })
                        ], style={'width': '33%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4('coût max : ', style={'display': 'inline-block', 'height' : '20px', 'width' : '80%' }),
                        html.Div(id='model_trigger_max', style={'display': 'inline-block', 'height' : '20px', 'width' : '19%' })
                        ], style={'width': '33%', 'display': 'inline-block'}),
                ], style={'width': '75%'}),
            ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3('Importance des variables', style={'textAlign': 'center'}),
                dcc.Graph(id='model_feat_importance')], style={'width': '50%', 'display': 'inline-block'}
            ),
            html.Div([
                html.H3('Variation du coût metier en fonction du seuil d\'acceptation du modèle', style={'textAlign': 'center'}),
                dcc.Graph(id='model_trigger_figure')], style={'width': '33%', 'display': 'inline-block'}
            ),
            html.Div([
                html.H3('Matrice de confusion du modèle en considérant le seuil optimal', style={'textAlign': 'center'}),
                dcc.Graph(id='model_trigger_figure_conf')], style={'width': '33%', 'display': 'inline-block'}
            ),
            html.Div([
                html.H3('Variation du score ROC et du nombre de Vrai/Faux Positif/Négatif en fonction du seuil d\'acceptation du modèle', style={'textAlign': 'center'}),
                dcc.Graph(id='model_trigger_figure_conf_var_true')], style={'width': '33%', 'display': 'inline-block'}
            ),
        ])

