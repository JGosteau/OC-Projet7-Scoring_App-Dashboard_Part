from dash import Input, Output, State, html, dash_table, dcc, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import dash_bootstrap_components as dbc
import requests
import pandas as pd
import numpy as np
import os, sys
import gc

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex


def register_callbacks(app, api_url = 'http://127.0.0.1:5000'):
    """
    Cette fonction gère tout les callbacks du Dashboard.
    """
    url = api_url + '/api/listcols'
    response = requests.get(url)
    qualcols = np.array(response.json()['qualcols'])

    url = api_url + '/api/uniquequalcols'
    response = requests.get(url)
    qual_list = pd.Series(response.json())

    url = api_url + '/api/ids'
    response = requests.get(url)
    ids_list = response.json()['ids']
    main_features = np.array(['AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'])
    
    @app.callback(
        Output('datainfo', 'data'),
        Input('datainfo', 'data'),
    )
    def get_data_info(datainfo) :
        """
        Récupère les informations du jeu de donnée d'entrainement : moyenne, medianne, quantile, ...
        """
        url = api_url + '/api/datainfo'
        response = requests.get(url)
        res_dict = response.json()
        conv_dict = {}
        for func in res_dict :
            conv_dict[func] = {}
            for var in res_dict[func] :
                for feat in res_dict[func][var] :
                    if feat in conv_dict[func] : 
                        conv_dict[func][feat][var] = res_dict[func][var][feat]
                    else :
                        conv_dict[func][feat] = {}
                        conv_dict[func][feat][var] = res_dict[func][var][feat]
        del response, res_dict
        gc.collect()
        return conv_dict

    @app.callback(
        Output('model_list', 'data'),
        Input('model_list', 'data')
        )
    def get_model_list(model_list):
        """
        Récupère la liste des modèles disponibles.
        """
        url = api_url + '/api/models'
        response = requests.get(url)
        df = response.json()['available models']
        return df
    

    @app.callback(
        Output("model_desc", "children"),
        Output("model_features", "data"),
        Output("model_feat_importance", "figure"),
        Output("model_roc", "children"),
        Input("model_dropdown", "value"),
        prevent_initial_call=True,
    )
    def get_info_model(value) :
        """
        Récupère les informations d'un modèle sélectionné dans la liste déroulante model_dropdown.
        """
        url = api_url + '/api/getinfomodel'
        response = requests.post(url, json={'model' : value})
        res = response.json()['description']
        features = response.json()['features']
        features_imp = response.json()['feature_importances']
        roc = '%.3f' %(float(response.json()['roc']))
        df = pd.DataFrame({'features' : features, 'feature importance' : features_imp})
        rows = []
        for k in res :
            row = html.Tr([html.Td(str(k)), html.Td(': ' + str(res[k]))])
            rows.append(row)
        table = html.Div(rows)

        fig = px.bar(df, x="feature importance", y="features")
        dfjson = df.to_json(date_format='iso', orient='split')
        return table, dfjson, fig, roc

    @app.callback(
        Output("model_trigger_figure", "figure"),
        Output("model_trigger_figure_conf", "figure"),
        Output("model_trigger_figure_conf_var_true", "figure"),
        Output("model_trigger_opt", "children"),
        Output("model_trigger_max", "children"),
        Output("model_trigger_var", "data"),
        Input("model_dropdown", "value"),
        Input("est_loan_rate2", "children"),
        prevent_initial_call=True,
    )
    def get_fig_trigger(model, loan_rate) :
        """
        Tracer les figures sur la variation de la fonction coût métier et des matrices de confusion en fonction du seuil.
        Détermine aussi le seuil optimal pour un taux d'emprunt en particulier (input est_loan_rate2)
        """

        if model is None or loan_rate is None:
            return go.Figure(),go.Figure(),go.Figure(),None,None, None
        loan_rate = float(loan_rate)
        url = api_url + '/api/trigger'
        #print('#### GETTING MODEL LIST - URL HOST : ', url)
        response = requests.post(url, json={'model' : model, 'loan_rate' : [loan_rate], 'reimb_ratio' : [0]}).json()
        df = pd.DataFrame(response['optimized_triggers'])
        loan_rate = np.ceil(loan_rate*100)/100
        color_seq = {
            0 : 'blue',
            0.05 : 'red',
            0.10 : 'green',
            0.15 : 'purple',
            0.20 : 'orange'
        }
        if loan_rate <= 1 :
            cost = df[df.loan_rate==loan_rate]['cost'].iloc[0]
            cost = "%.3f" %(cost)
            opt_trigger = df[df.loan_rate==loan_rate]['opt_trigger']
            opt_trigger = "%.2f" %(opt_trigger)
        else :
            cost = 'le taux doit être inf à 1'
            opt_trigger = 'le taux doit être inf à 1'
        graph_data = pd.DataFrame(response['exp_cost_func'])
        TN = graph_data
        
        color_seq[loan_rate] = 'black'
        fig1 = px.line(graph_data[graph_data.loan_rate.isin(list(color_seq.keys()))], x='trigger', y='cost', color='loan_rate', color_discrete_map=color_seq)
        dfjson = graph_data[graph_data.loan_rate == loan_rate].to_json(date_format='iso', orient='split')
        trig = df[df.loan_rate==loan_rate]['opt_trigger'].iloc[0]
        tmp = graph_data[(graph_data.loan_rate == loan_rate) & (np.round(graph_data.trigger,decimals=2) == trig)][['TP','TN']]

        TP, TN = tmp.iloc[0]

        FP = 1-TP
        FN = 1-TN

        data = pd.DataFrame([[TP,FP],[FN,TN]], columns = ['Acc.', 'Non Acc.'], index=['Rembourse', 'Rembourse Pas'])
        data = data.apply(lambda x : np.round(x*100))
        fig2 = px.imshow(data, text_auto=True)


        df_T = graph_data[graph_data.loan_rate == 0].set_index('trigger')[['TP', 'TN']]
        df_T['FP'] = 1-df_T['TP']
        df_T['FN'] = 1-df_T['TN']
        df_T.columns = ['VP', 'FN', 'FP', 'VN']
        df_T = df_T.stack().reset_index()
        df_T.columns = ['trigger', 'status', 'value']
        fig3 = px.line(df_T, x='trigger', y='value', color = 'status', color_discrete_sequence=['green', 'red','green', 'red'])
        fig3.data[-2]['line']['dash'] = 'dash'
        fig3.data[-1]['line']['dash'] = 'dash'
        fig3.add_vline(x=trig)
        return fig1, fig2, fig3, opt_trigger, cost, dfjson
    

    @app.callback(
        Output("id_value", "value"),
        Output("id_value2", "value"),
        Input("id_value", "value"),
        Input("id_value2", "value"),
    )
    def update_options(value1, value2):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if  trigger_id == "id_value" :
            value = value1
        elif trigger_id == "id_value2" :
            value = value2
        return value, value
        
    @app.callback(
        Output("Imputer_method", "value"),
        Output("Imputer_method2", "value"),
        Input("Imputer_method", "value"),
        Input("Imputer_method2", "value"),
    )
    def update_options(value1, value2):
        """
        Gère le choix de la méthode d'imputation sur tout les onglets
        """
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if  trigger_id == "Imputer_method" :
            value = value1
        elif trigger_id == "Imputer_method2" :
            value = value2
        return value, value

    @app.callback(
        Output("model_dropdown", "options"),
        Output("model_dropdown2", "options"),
        Output("model_dropdown3", "options"),
        Input("model_dropdown", "search_value"),
        State("model_dropdown", "value"),
        Input("model_dropdown2", "search_value"),
        State("model_dropdown2", "value"),
        Input("model_dropdown3", "search_value"),
        State("model_dropdown3", "value"),
        Input('model_list', 'data'),
    )
    def update_options(search_value1,value1,search_value2,value2,search_value3,value3,model_list):
        """
        Gère le choix du modèle sur tout les onglets
        """
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if  trigger_id == "model_dropdown" :
            value = value1
        elif trigger_id == "model_dropdown1" :
            value = value2
        else :
            value = value3
        list_options = model_list
        if value is None :
            value = model_list[0]
        
        return list_options,list_options,list_options

    @app.callback(
        Output("model_dropdown", "value"),
        Output("model_dropdown2", "value"),
        Output("model_dropdown3", "value"),
        Input("model_dropdown", "value"),
        Input("model_dropdown2", "value"),
        Input("model_dropdown3", "value"),
        Input('model_list', 'data'),
    )   
    def update_options(value1,value2,value3,model_list):
        """
        Gère le choix du modèle sur tout les onglets
        """
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "model_dropdown" :
            value = value1
        elif trigger_id == "model_dropdown2" :
            value = value2
        else :
            value = value3
        if value is None :
            value = model_list[0]
        
        return value, value, value

    @app.callback(
        Output("est_loan_rate", "value"),
        Output("est_loan_rate2", "children"),
        Output("est_time", "children"),
        Output("input_AMT_CREDIT", "value"),
        Output("max_gain", "children"),
        Input("input_AMT_CREDIT", "value"),
        Input("input_AMT_GOODS_PRICE", "value"),
        Input("input_AMT_ANNUITY", "value"),
        Input("est_loan_rate", "value"),
        prevent_initial_call=True,
    )
    def estimate_loan_time(credit, loan, annuity, loan_rate) :
        """
        Détermine le taux d'emprunt en fonction de l'emprunt et du crédit.
        """
        est_time = None
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if loan is not None :
            if trigger_id in ['input_AMT_CREDIT', "input_AMT_GOODS_PRICE", "input_AMT_ANNUITY"] :
                loan_rate = '%.2f' %(credit/loan-1)
            elif trigger_id == 'est_loan_rate' :
                credit = loan*(1+loan_rate)
            est_time = '%.2f' %(credit/annuity)
            benef_max =  '%d $' %(credit-loan)
        else :
            loan_rate = 0  
            benef_max = None           
        return loan_rate,loan_rate, est_time, credit, benef_max



    @app.callback(
        Output("main_features", "children"),
        Output("model_features_box", "children"),
        Input("model_features", "data"),
        Input("Imputer_method", "value"),
        Input("id_value", "value"),
        prevent_initial_call=True,
    )
    def get_desc(dfjson, imputer_method, id):
        """
        Récupère la liste des variables d'un modèle et permet le choix de la valeur de ces variables dans l'onglet 'Info Individu'
        """
        df = pd.read_json(dfjson, orient='split')
        features = df.features
        df = df.set_index('features')['feature importance']
        norm_df = df.apply(lambda x : (x-min(df))/(max(df)-min(df)))
        
        cmaps = colormaps['rainbow']
        color_df = norm_df.apply(lambda x : to_hex(cmaps(x)))
        features_values = {k : None for k in features}
        for feat in main_features :
            features_values[feat] = None

        if id in ids_list :
            url = api_url + '/api/getinfoid'
            response = requests.post(url, json={"SK_ID_CURR" : id}).json()
            for k in features_values :
                features_values[k] = response[k]
        if imputer_method != 'None' :
            url = api_url + '/api/imputer'
            response = requests.post(url, json={'imputer' : imputer_method, 'x' : {}}).json()
            for k in features_values :
                if features_values[k] is None :
                    features_values[k] = response[k]
                elif type(features_values[k]) != str :
                    if np.isnan(features_values[k]) :
                        features_values[k] = response[k]


        divs_features = []
        for feat in main_features :
            input_name = "input_%s" %(feat) 
            value = features_values[feat]
            if feat in qualcols :
                input_field = dcc.Dropdown(list(qual_list[feat]), value=value, id = input_name, style={'width' : '75%'})
            else :
                input_field =  dcc.Input(id=input_name, type="number", value=value, style={'width' : '75%'}, debounce = True)
            if feat in df.index :
                text_fi = "(%.3f)" %(df[feat])
                color_fi = color_df[feat]
            else :
                text_fi = None
                color_fi = 'black'
            div = html.Div([
                html.H6(feat, style={'width' : '69%', 'display': 'inline-block'}),
                html.I(text_fi, style={'width' : '30%', 'textAlign' : 'right', 'color' : color_fi}),
                input_field
            ], style={'width' : '15%', 'display': 'inline-block', 'vertical-align': 'top'})
            divs_features.append(div)
        main_divs = divs_features

        
        # Secondary Features
        n_feat_cols = 3
        n_feat_rows = 3

        tabs = []
        new_features = np.array(features)[~np.isin(features, main_features)]

        n_tabs = int(np.ceil(len(new_features)/(n_feat_cols*n_feat_rows)))
        for j in range(n_tabs) :
            ini = j*n_feat_cols*n_feat_rows
            end = (j+1)*n_feat_cols*n_feat_rows
            divs_features = []
            for feat in new_features[ini : end] :
                input_name = "input_%s" %(feat)
                value = features_values[feat]
                if feat in qualcols :
                    input_field = dcc.Dropdown(list(qual_list[feat]), value=value, id = input_name, style={'width' : '75%'})
                else :
                    input_field =  dcc.Input(id=input_name, type="number", value=value, style={'width' : '75%'}, debounce = True)
                div = html.Div([
                    html.H6(feat, style={'width' : '69%', 'display': 'inline-block'}),
                    html.I("(%.3f)" %(df[feat]), style={'width' : '30%', 'textAlign' : 'right', 'color' : color_df[feat]}),
                    input_field
                ], style={'width' : '%s%%' %(99/n_feat_cols), 'display': 'inline-block', 'vertical-align': 'top'})
                divs_features.append(div)
            if j != n_tabs-1 :
                text = 'Var %d-%d' %(ini, end)
            else : 
                text = 'Var %d-%d' %(ini, len(new_features))
            divs_features = dcc.Tab(label = text, children = divs_features, style={'width' : '100%', 'height' : '100px'})
            tabs.append(divs_features)
        tabs = dcc.Tabs(tabs,id='tabs_features')
        return main_divs, tabs

    @app.callback(
        Output('indiv_data', 'data'),
        Input('predict', 'n_clicks'),
        Input('act_data', 'n_clicks'),
        State("input_AMT_GOODS_PRICE", "value"),
        State("input_AMT_CREDIT", "value"),
        State("input_AMT_ANNUITY", "value"),
        State("input_AMT_INCOME_TOTAL", "value"),
        State("model_features_box", "children"),
    )
    def update_indiv_data(n_clicks, n_clicks2, loan, credit, annuity, income, features_box):
        """
        Stocke dans une variable les données d'un individu
        """
        x = {}
        x['AMT_GOODS_PRICE'] = loan
        x['AMT_CREDIT'] = credit
        x['AMT_ANNUITY'] = annuity
        x['AMT_INCOME_TOTAL'] = income
        tabs = features_box['props']['children']
        for tab in tabs :
            divs = tab['props']['children']
            for div in divs :
                feat = div['props']['children'][0]['props']['children']
                try :
                    value = div['props']['children'][-1]['props']['value']
                except :
                    value = None
                x[feat] = value
        print(x)
        return x
    
    @app.callback(
        Output('indiv_tabs', 'data'),
        Input('predict', 'n_clicks'),
        Input('act_data', 'n_clicks'),
        State("input_AMT_GOODS_PRICE", "value"),
        State("input_AMT_CREDIT", "value"),
        State("input_AMT_ANNUITY", "value"),
        State("input_AMT_INCOME_TOTAL", "value"),
        Input("tabs_features", "children"),
        Input("tabs_features", "value"),
    )
    def update_indiv_data(n_click1, n_click2, loan, credit, annuity, income,tabs, active_tab):
        """
        Stocke dans une variable les données d'un individu pour la plage de variables séléctionnées.
        """
        x = {}
        x['AMT_GOODS_PRICE'] = loan
        x['AMT_CREDIT'] = credit
        x['AMT_ANNUITY'] = annuity
        x['AMT_INCOME_TOTAL'] = income
        tabs = tabs
        if active_tab is None :
            active_tab = 0
        else :
            active_tab = int(active_tab.split('-')[-1])-1
        print(active_tab)
        tab = tabs[active_tab]
        divs = tab['props']['children']
        for div in divs :
            feat = div['props']['children'][0]['props']['children']
            try :
                value = div['props']['children'][-1]['props']['value']
            except :
                value = None
            x[feat] = value
        return x


    @app.callback(
        Output('polar_graph_feat', 'figure'),
        Output('boxplot_graph_feat', 'figure'),
        Input('model_dropdown', 'value'),
        Input('id_value', 'value'),
        Input('datainfo', 'data'),
        Input('indiv_tabs', 'data'),
        prevent_initial_call=True,
    )
    def get_polar_graph(model, id, datainfo, x) :
        """
        Trace les figures polaires et en boxplot pour la comparaison des valeurs des variables d'un individu par rapport à la population du jeu d'entrainement.
        """
        none_res =go.Figure(),go.Figure()
        data = []
        columns = []
        for k in x :
            if k not in qualcols :
                data.append(x[k])
                columns.append(k)


        df = pd.DataFrame([data], index = ['individu'], columns=columns)
        df_mean = pd.DataFrame(datainfo['mean'])[columns]
        df_median = pd.DataFrame(datainfo['median'])[columns].loc[['all']]
        df_q3 = pd.DataFrame(datainfo['q3'])[columns].loc[['all']]
        df_q1 = pd.DataFrame(datainfo['q1'])[columns].loc[['all']]
        df_d9 = pd.DataFrame(datainfo['d9'])[columns].loc[['all']]
        df_d1 = pd.DataFrame(datainfo['d1'])[columns].loc[['all']]

        df_min = pd.DataFrame(datainfo['min'])[columns].loc['all']
        df_max = pd.DataFrame(datainfo['max'])[columns].loc['all']
        df_iqr = df_q3 - df_q1
        df_iqr_l = df_q1 - 1.5*df_iqr
        df_iqr_h = df_q3 + 1.5*df_iqr
        df_iqr_l.index = ['iqr_l']
        df_iqr_h.index = ['iqr_h']
        print(df_mean.loc[['all']].index)
        df = pd.concat((df, df_mean.loc[['all']])).reset_index()
        df['index'][df['index'] == 'all'] = 'moyenne'
        df = df.set_index('index')

        df_norm = (df-df_min)/(df_max-df_min)
        
        dff = df.unstack()
        dff_norm = df_norm.unstack()
        dff = pd.DataFrame({'values' : dff, 'norm' : dff_norm}).reset_index()

        dff.columns = ['features', 'level','values', 'norm']
        fig = px.line_polar(dff, r="norm", theta="features", custom_data=['values'], color='level', line_close=True, range_r=[0, max(dff.norm)])
        fig.update_traces(
            hovertemplate="<br>".join([
                "norm. values: %{r}",
                "features: %{theta}",
                "values: %{customdata[0]}",
            ])
        )

        fig_boxplot = make_subplots(rows=int(np.ceil(len(columns)/2))+1, cols=2, horizontal_spacing = 0.4)
        for i, feat in enumerate(columns) :
            if i > len(columns)/2 :
                c = 2
                di = int(np.ceil(len(columns)/2))
            else :
                c= 1
                di = 0
            data = go.Box(
                q1=df_q1[[feat]].iloc[0], 
                median=df_median[[feat]].iloc[0],
                q3=df_q3[[feat]].iloc[0], 
                lowerfence=df_d1[[feat]].iloc[0],
                upperfence=df_d9[[feat]].iloc[0], 
                y=[feat],
                marker_color  = 'red',
                name='all',
            )
            data2 = go.Scatter(
                x = df[[feat]].iloc[0],
                y=[feat],
                marker_color = 'black',
                mode = 'markers',
                name = 'indiv',
            )
            fig_boxplot.add_trace(data, row = i+1-di, col=c)
            fig_boxplot.add_trace(data2, row = i+1-di, col=c)
        fig_boxplot.update_layout(showlegend=False, margin=dict(l=100, r=0, t=0, b=0))
        return fig, fig_boxplot
        
    @app.callback(
        Output("prediction_value", "children"),
        Output("prediction_gain", "children"),
        Output("prediction_gain", "style"),
        Output("ratio_gain", "children"),
        Output("ratio_gain", "style"),
        Output("waterfall_figure", "figure"),
        Output("prediction_status", "children"),
        Output("prediction_status", "style"),
        Output("pred_calc_status", "children"),
        Input('indiv_data', 'data'),
        Input('predict', 'n_clicks'),
        Input("model_dropdown", "value"),
        State("model_trigger_var", "data"),
        State("model_trigger_opt", "children"),
        prevent_initial_call=True,
    )
    def proba_calc(x, n_clicks, model, dfjson, model_trigger_opt) :
        """
        Calcul la probabilité de remboursement d'un individu et trace la figure des contributions des différentes varaibles du modèle sur cette prédiction.
        """
        style_status = {'width' : '20%', 'display': 'inline-block'}
        style_gain = {'width' : '20%', 'display': 'inline-block'}
        none_res = [None, None, style_gain, None, style_gain, go.Figure(), None, style_status, None]

        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == 'model_dropdown' :
            return none_res
        if trigger_id == 'predict' :
            print('Callback predict')
            loan = x['AMT_GOODS_PRICE']
            credit = x['AMT_CREDIT']
            annuity = x['AMT_ANNUITY']
            income = x['AMT_INCOME_TOTAL']
            sent = {
                "model" : model,
                "x" : x
            }
            if None in x.values() :
                print('none values in data')
                for k in x :
                    if x[k] is None :
                        none_res[-1] = 'Erreur ! Valeur nulle pour la variable : %s' %(k)
                return none_res
            print(str(sent).replace('\'','"').replace(',',',\n').replace('{','{\n').replace('}','\n}'))
            trigger_var_df = pd.read_json(dfjson, orient='split')
            
            opt_trigger = float(model_trigger_opt)
            url = api_url + '/api/predict'
            #print('#### GETTING MODEL LIST - URL HOST : ', url)
            response = requests.post(url, json=sent)
            probability = response.json()['probability']
            contribs = pd.DataFrame(response.json()['contribs'])
            
            fig = go.Figure(go.Waterfall(
                name = "Prediction", #orientation = "h", 
                measure = ["relative"] * (len(contribs)-1) + ["total"],
                x = contribs.index,
                y = contribs.Contributions,
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            gain = probability*(credit-loan) + (probability-1) * loan
            gain = probability*(credit) - loan
            loan_rate = credit/loan - 1
            gain = loan*(loan_rate*probability-(1-probability))

            if probability >= opt_trigger :
                cost = max(trigger_var_df['cost'])
                gain = loan*cost
                status = 'Accepté'
                status_color = 'green'
                probability = "%.2f (>%.2f)" %(probability, opt_trigger)
            else :
                cost = trigger_var_df[trigger_var_df['trigger']>= probability]['cost'].iloc[0]
                gain = loan*cost
                status = 'Rejeté'
                status_color = 'red'
                probability = "%.2f (<%.2f)" %(probability, opt_trigger)
            cmaps = colormaps['brg']
            if credit == loan :
                ratio_gain = np.inf
                color_gain = cmaps(0.5)
            else :
                ratio_gain = gain/(credit-loan)
                color_gain = cmaps(ratio_gain/2+0.5)
            gain = '%d $' %(gain)
            ratio_gain = '%.2f' %(ratio_gain)
            style_status['color'] = status_color
            style_gain['color'] = to_hex(color_gain)
            return probability, gain, style_gain, ratio_gain, style_gain, fig, status, style_status, 'OK'