from dash import Input, Output, State, html, dash_table, dcc, callback_context
import plotly.express as px
import plotly.graph_objects as go
#import dash_bootstrap_components as dbc
import requests
import pandas as pd
import numpy as np
import os, sys

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex


def register_callbacks(app, api_html = 'http://127.0.0.1:5000'):
    #xtrain = pd.read_csv(os.path.join(os.path.dirname(__file__),"..", "..", "model", "data", 'xtrain_model.csv'), compression='gzip')
    #qualcols = np.load(os.path.join(os.path.dirname(__file__),"..", "..", "model", "data", 'qualcols.npy'), allow_pickle=True)

    #qual_list = xtrain[qualcols].apply(np.unique)
    url = api_html + '/api/listcols'
    response = requests.get(url)
    qualcols = np.array(response.json()['qualcols'])
    print(qualcols)

    url = api_html + '/api/uniquequalcols'
    response = requests.get(url)
    qual_list = pd.Series(response.json())
    print(qual_list)

    url = api_html + '/api/ids'
    response = requests.get(url)
    ids_list = response.json()['ids']
    main_features = np.array(['AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'])

    

    @app.callback(
        Output('model_list', 'data'),
        Input('model_list', 'data')
        )
    def get_model_list(model_list):
        # some expensive data processing step
        url = api_html + '/api/models'
        
        #print('#### GETTING MODEL LIST - URL HOST : ', url)
        response = requests.get(url)
        df = response.json()['available models']
        return df
    
    @app.callback(
        Output('model_features_box', 'style'),
        Input('hide_sec_features_button', 'n_clicks'),
        State('model_features_box', 'style'),
        prevent_initial_call=True,
    )
    def hide(n_clicks, style):
        if 'display' in style.keys() :
            if style['display'] == 'none' :
                style['display'] = 'block'
            else :
                style['display'] = 'none'
        else :
            style['display'] = 'none'
        return style

    @app.callback(
        Output("model_desc", "children"),
        Output("model_features", "data"),
        Output("model_feat_importance", "figure"),
        Input("model_dropdown", "value"),
        prevent_initial_call=True,
    )
    def get_info_model(value) :
        url = api_html + '/api/getinfomodel'
        #print('#### GETTING MODEL LIST - URL HOST : ', url)
        response = requests.post(url, json={'model' : value})
        res = response.json()['description']
        features = response.json()['features']
        features_imp = response.json()['feature_importances']
        df = pd.DataFrame({'features' : features, 'feature importance' : features_imp})
        rows = []
        for k in res :
            row = html.Tr([html.Td(str(k)), html.Td(': ' + str(res[k]))])
            rows.append(row)
        table = html.Div(rows)

        fig = px.bar(df, x="feature importance", y="features")
        dfjson = df.to_json(date_format='iso', orient='split')
        return table, dfjson, fig

    @app.callback(
        Output("model_trigger_figure", "figure"),
        Output("model_trigger_figure_conf", "figure"),
        Output("model_trigger_figure_conf_var_true", "figure"),
        Output("model_trigger_opt", "children"),
        Output("model_trigger_max", "children"),
        Output("model_trigger_var", "data"),
        Input("model_dropdown", "value"),
        Input("est_loan_rate2", "children"),
        #Input("est_loan_rate", "value"),
        prevent_initial_call=True,
    )
    def get_fig_trigger(model, loan_rate) :
        if model is None or loan_rate is None:
            return go.Figure(),go.Figure(),go.Figure(),None,None, None
        loan_rate = float(loan_rate)
        url = api_html + '/api/trigger'
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
            #cost = min(df['opt_trigger'])
            opt_trigger = 'le taux doit être inf à 1'
            #opt_trigger = min(df['opt_trigger'])
            #cost = df[df.opt_trigger==opt_trigger]['cost'].iloc[0]
            #opt_trigger = "%.2f" %(opt_trigger)
            #cost = "%.3f" %(cost)
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


        df_T = graph_data[graph_data.loan_rate == 0].set_index('trigger')[['ROC', 'TP', 'TN']]
        df_T['FP'] = 1-df_T['TP']
        df_T['FN'] = 1-df_T['TN']
        df_T.columns = ['Score ROC', 'Rembourse et Acc.', 'Rembourse Pas et Non Acc.', 'Rembourse et Non Acc.', 'Rembourse Pas et Acc.']
        df_T = df_T.stack().reset_index()
        df_T.columns = ['trigger', 'status', 'value']
        fig3 = px.line(df_T, x='trigger', y='value', color = 'status', color_discrete_sequence=['black','green', 'red','green', 'red'])
        fig3.data[-2]['line']['dash'] = 'dash'
        fig3.data[-1]['line']['dash'] = 'dash'
        fig3.add_vline(x=trig)
        return fig1, fig2, fig3, opt_trigger, cost, dfjson
    



    @app.callback(
        Output("model_dropdown", "options"),
        Output("model_dropdown2", "options"),
        #Output("model_dropdown", "value"),
        #Output("model_dropdown2", "value"),
        Input("model_dropdown", "search_value"),
        State("model_dropdown", "value"),
        Input("model_dropdown2", "search_value"),
        State("model_dropdown2", "value"),
        Input('model_list', 'data'),
    )
    def update_options(search_value1,value1,search_value2,value2,model_list):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        value = value1 if trigger_id == "model_dropdown" else value2
        list_options = model_list
        if value is None :
            value = model_list[0]
        
        return list_options,list_options#, value, value

    @app.callback(
        Output("model_dropdown", "value"),
        Output("model_dropdown2", "value"),
        Input("model_dropdown", "value"),
        Input("model_dropdown2", "value"),
        Input('model_list', 'data'),
    )   
    def update_options(value1,value2,model_list):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        value = value1 if trigger_id == "model_dropdown" else value2
        list_options = model_list
        if value is None :
            value = model_list[0]
        
        return value, value

    """@app.callback(
        Output("est_loan_rate", "value"),
        Output("est_loan_rate2", "children"),
        Output("est_time", "children"),
        Input("input_AMT_GOODS_PRICE", "value"),
        Input("input_AMT_CREDIT", "value"),
        Input("input_AMT_ANNUITY", "value"),
        #Input("est_calcul", "n_clicks"),
        #prevent_initial_call=True,
    )
    def estimate_loan_time(loan, credit, annuity) :
        est_loan_rate = 0
        est_time = None
        try :
            est_loan_rate = '%.2f' %(credit/loan-1)
            est_time = '%.2f' %(credit/annuity)
        except :
            None
        return est_loan_rate,est_loan_rate, est_time"""

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
        #Input("est_calcul", "n_clicks"),
        prevent_initial_call=True,
    )
    def estimate_loan_time(credit, loan, annuity, loan_rate) :
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
        #Input('reset', 'n_clicks'),
        Input("Imputer_method", "value"),
        Input("id_value", "value"),
        prevent_initial_call=True,
    )
    def get_desc(dfjson, imputer_method, id):
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
            url = api_html + '/api/getinfoid'
            response = requests.post(url, json={"SK_ID_CURR" : id}).json()
            for k in features_values :
                features_values[k] = response[k]
        if imputer_method != 'None' :
            url = api_html + '/api/imputer'
            response = requests.post(url, json={'imputer' : imputer_method, 'x' : {}}).json()
            for k in features_values :
                if features_values[k] is None :
                    features_values[k] = response[k]
                elif type(features_values[k]) != str :
                    if np.isnan(features_values[k]) :
                        features_values[k] = response[k]
        #print(tmp)
        #table_header = [html.Thead(html.Tr([html.Th("Opération"), html.Th("Fonction")]))]
        


        # Main Features 

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
        n_feat_cols = 6
        n_feat_rows = 2

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
                ], style={'width' : '15%', 'display': 'inline-block', 'vertical-align': 'top'})
                #divs_features += div
                divs_features.append(div)
            if j != n_tabs-1 :
                text = 'Var %d-%d' %(ini, end)
            else : 
                text = 'Var %d-%d' %(ini, len(new_features))
            divs_features = dcc.Tab(label = text, children = divs_features, style={'width' : '100%', 'height' : '100px'})
            tabs.append(divs_features)
        tabs = dcc.Tabs(tabs)
        return main_divs, tabs



    @app.callback(
        Output("prediction_value", "children"),
        Output("prediction_gain", "children"),
        Output("prediction_gain", "style"),
        Output("ratio_gain", "children"),
        Output("ratio_gain", "style"),
        #Output("contribs", "data"),
        Output("waterfall_figure", "figure"),
        Output("prediction_status", "children"),
        Output("prediction_status", "style"),
        Input('predict', 'n_clicks'),
        Input("model_dropdown", "value"),
        Input("model_features_box", "children"),
        Input("input_AMT_GOODS_PRICE", "value"),
        Input("input_AMT_CREDIT", "value"),
        Input("input_AMT_ANNUITY", "value"),
        Input("input_AMT_INCOME_TOTAL", "value"),
        Input("model_trigger_var", "data"),
        Input("model_trigger_opt", "children"),
        prevent_initial_call=True,
    )
    def test(n_clicks, model, children, loan, credit, annuity, income, dfjson, model_trigger_opt) :

        style_status = {'width' : '20%', 'display': 'inline-block'}
        style_gain = {'width' : '20%', 'display': 'inline-block'}

        #print(children['props']['children'][0]['props']['children'][1]['props']['value'])
        none_res = (None, None, style_gain, None, style_gain, go.Figure(), None, style_status)
        try :
            tabs = children['props']['children']
        except : 
            return none_res
        #print(children['props']['children'][1]['props']['children'][0]['props']['children'][1])
        x = {}
        x['AMT_GOODS_PRICE'] = loan
        x['AMT_CREDIT'] = credit
        x['AMT_ANNUITY'] = annuity
        x['AMT_INCOME_TOTAL'] = income

        for tab in tabs :
            divs = tab['props']['children']
            for div in divs :
                feat = div['props']['children'][0]['props']['children']
                try :
                    value = div['props']['children'][-1]['props']['value']
                except :
                    value = None
                x[feat] = value
        sent = {
            "model" : model,
            "x" : x
        }
        print(x['AMT_CREDIT'])
        if None in x.values() :
            return none_res
        trigger_var_df = pd.read_json(dfjson, orient='split')
        opt_trigger = float(model_trigger_opt)
        url = api_html + '/api/predict'
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
        #contribs.to_json(date_format='iso', orient='split')
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
        return probability, gain, style_gain, ratio_gain, style_gain, fig, status, style_status


    