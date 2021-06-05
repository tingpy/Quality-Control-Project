# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:08:52 2021

@author: acer
"""


import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_table

import pandas as pd
import numpy as np
import datetime
import json

<<<<<<< HEAD
# import sys
# sys.path.append("C:\\Users\\acer\\Desktop\\Chimei\\QC data\\app_structure")
=======

>>>>>>> 5c2837210fe9402e90ad4ef347bc0726b575c686
from app_structure import function

# import all pages in the app folder
from app_structure import Navi_bar, Import_New_Data, EDA, Manage_Data, Model, mainpage
from app_structure.quality_py import deal, spec, agent, customer


import os
os.chdir("./")
<<<<<<< HEAD

# prepare for the dataframe
deal = 0
spec = 0
agent = 0
customer = 0


import os
# os.chdir('C:\\Users\\acer\\Desktop\\Chimei\\QC data')
# local_main = 'C:\\Users\\acer\\Desktop\\Chimei\\QC data'
=======
local_main = "./"
>>>>>>> 5c2837210fe9402e90ad4ef347bc0726b575c686
stan_result = []

# needed only if running this as a single page app
# external_stylesheets = ['C:\\Users\\acer\\Desktop\\Chimei\\QC data\\assets\\bootstrap.min.css'] # dbc.themes.LUX

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# change to app.layout if running as single page app instead
app.layout = html.Div([
    
    html.Div(id='intermediate_store',
             children=[
                  html.Div(id='intermediate_layer'),
                  html.Div(id='intermediate_layer2'),
                  dcc.Store(id='intermediate_layer3',
                            storage_type='session'),
                  dcc.Store(id='intermediate_layer4',
                            storage_type='session'),
                  dcc.Store(id='intermediate_layer5',
                            storage_type='session'),
                  dcc.Store(id='info_store1',
                            storage_type='session'),
                  dcc.Store(id='info_store2',
                            storage_type='session'),
                  dcc.Store(id='purchase_info',
                            storage_type='session'),
                  dcc.Store(id='focus_spec_intermediate',
                            storage_type='memory'),
                  dcc.Store(id='rate_and_score_info',
                            storage_type='memory'),
                  dcc.Store(id='total_ranking_store',
                            storage_type='memory'),
                 ]),
    
    # for the navbar
    dcc.Location(id='url', refresh=False),
    Navi_bar.navbar,
    html.Div(id='page-content'),
    # dbc.Container(,
    #             color="dark",
    #             dark=True,
    #             className="mb-4",)

])

######## Callback for NavBar ###############################
# for NavBar
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/Import_New_Data':
        return Import_New_Data.layout
    elif pathname == '/EDA':
        return EDA.layout
    elif pathname == '/Model':
        return Model.layout
    elif pathname == '/Manage_Data':
        return Manage_Data.layout
    else:
        return mainpage.layout
###########################################################
        
######## Callback for mainpage ############################
@app.callback([Output('loc_info', 'children'),
               Output('dir_button', 'n_clicks')],
              [Input('input_loc', 'value'),
               Input('dir_button', 'n_clicks')])

def show_directory(loc, n_click):
    print(loc)
    if n_click > 0:
        valid = False
        if 'QC data' in loc:
            valid = True
        
        if valid:
            children = [
                dbc.Row(html.P(children = 'Your directory will be set as ' + loc)),
                dbc.Row(html.P(children = 'App will start to import data, please wait~'))]
            return children, 1
        else:
            return [], 0
    else:
        raise PreventUpdate


@app.callback([Output('loc_warning', 'displayed'),
               Output('loc_confirm', 'displayed'),
               Output('show_next_step', 'style')],
              [Input('input_loc', 'value'),
               Input('dir_button', 'n_clicks')])

def show_loc_warning(loc, n_click):
    if n_click > 0:
        if 'QC data' in loc:
            loc_new = loc.replace('\\', '/')
            os.chdir(loc_new) 
            
            global spec, deal, agent, customer
            spec, deal, agent, customer = function.import_data(loc)
            return False, False, {'display': 'block'}
        elif 'QC data' not in loc:
            return True, False, {'display': 'none'}
        else:
            return False, True, {'display': 'none'}
    else:
        raise PreventUpdate
    

@app.callback(Output('confirm', 'displayed'),
              Input('dropdown', 'value'))
def display_confirm(value):
    if value == 'Danger!!':
        return True
    return False
    
######## Callback for Import_New_Data #####################

# import new deal data
@app.callback(Output('output_deal_inform', 'children'),
              [Input('upload_data_deal', 'contents')],
              [State('upload_data_deal', 'filename'),
               State('upload_data_deal', 'last_modified')])
def update_output_deal(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None: 
       try:
           children = [
           Import_New_Data.parse_contents(c, n, d, deal, 'deal') for c, n, d in
           zip([list_of_contents], [list_of_names], [list_of_dates])]
       except:
           children = [
           Import_New_Data.parse_contents(c, n, d, deal, 'deal') for c, n, d in
           zip(list_of_contents, list_of_names, list_of_dates)]
            
    return children

# import new spec data
@app.callback(Output('output_spec_inform', 'children'),
              [Input('upload_data_spec', 'contents')],
              [State('upload_data_spec', 'filename'),
               State('upload_data_spec', 'last_modified')])
def update_output_spec(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None: 
        try:
            children = [
            Import_New_Data.parse_contents(c, n, d, spec, 'spec') for c, n, d in
            zip([list_of_contents], [list_of_names], [list_of_dates])]
        except:
            children = [
            Import_New_Data.parse_contents(c, n, d, spec, 'spec') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
            
    return children

# import new agent data
@app.callback(Output('output_agent_inform', 'children'),
          [Input('upload_data_agent', 'contents')],
          [State('upload_data_agent', 'filename'),
            State('upload_data_agent', 'last_modified')])
def update_output_agent(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None: 
        try:
            children = [
            Import_New_Data.parse_contents(c, n, d, spec, 'agent') for c, n, d in
            zip([list_of_contents], [list_of_names], [list_of_dates])]
        except:
            children = [
            Import_New_Data.parse_contents(c, n, d, spec, 'agent') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
            
    return children

# import new customer data
@app.callback(Output('output_customer_inform', 'children'),
          [Input('upload_data_agent', 'contents')],
          [State('upload_data_agent', 'filename'),
            State('upload_data_agent', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None: 
        try:
            children = [
            Import_New_Data.parse_contents(c, n, d, customer, 'customer') for c, n, d in
            zip([list_of_contents], [list_of_names], [list_of_dates])]
        except:
            children = [
            Import_New_Data.parse_contents(c, n, d, customer, 'customer') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
            
    return children
###########################################################

######## Callback for EDA #################################
# create a new folder for each material and change directory
@app.callback(Output('intermediate_layer5', 'data'),
              Input('material_dropdown', 'value'))

def create_folder(select_material):
    try:
        os.chdir('./' + select_material)
    except:
        os.mkdir('./' + select_material)
        os.chdir('./' + select_material)
  
    return select_material

# material dropdown
@app.callback([Output('spec_inform_update', 'children'),
               Output('select_spec_EDA', 'options'),
               Output('select_spec_EDA', 'value'),
               Output('focus_spec_var', 'children'),
               Output('intermediate_layer', 'value')],
              [Input('material_dropdown', 'value'),
               Input('material_slider', 'value')])

def update_spec_info(select_material, select_date):
    df = spec[spec['material'] == select_material]
    unique_spec = df['spec name'].unique()
    
    spec_df = pd.DataFrame(columns=['Spec Name', 'First Day', 'Last Day'])
    row_cnt = 0
    for i in unique_spec:
        temp_df = df[df['spec name'] == i]
        spec_df.loc[row_cnt] = [i, temp_df['date'].min(), temp_df['date'].max()]
        row_cnt = row_cnt + 1
    
    spec_df = spec_df.sort_values('Last Day', ascending=False)
    
    end_date = datetime.datetime(select_date[1]+1, 1, 1)
    end_bool = (spec_df['Last Day'] < end_date).values
    begin_date = datetime.datetime(select_date[0]-1, 12, 31)
    begin_bool = np.logical_and((spec_df['First Day'] < begin_date).values,
                                (spec_df['Last Day'] > begin_date).values)
    
    
    spec_df = spec_df.loc[np.logical_and(end_bool, begin_bool),:]
    
    # for spec_inform_update    
    children = dash_table.DataTable(
                  data=spec_df.to_dict('records'),
                  columns=[{'name': i, 'id': i} for i in spec_df.columns])
    
    alive_spec = spec_df['Spec Name'].values
    options = [{'label': i, 'value': i} for i in alive_spec]
    
    # for focus_spec_var
    children2 = EDA.create_focus_spec_dropdown(select_material, options)
    
    return children, options, alive_spec[0], children2, alive_spec

# update the graph for the selected spec
@app.callback(Output('spec_EDA_graph', 'figure'),
              [Input('material_dropdown', 'value'),
                Input('select_spec_EDA', 'value'),
                Input('table_type', 'value'),
                Input('data_type', 'value')])

def update_spec_graph(select_material, select_spec, table_type, data_type):

    df = spec[spec['material'] == select_material]
    
    return EDA.create_spec_plot(df, select_spec, table_type, data_type)

# send the message that you have chosen some focus spec
@app.callback(Output('choose_spec_intermediate', 'children'),
              [Input('confirm_spec_button', 'n_clicks'),
                Input('choose_focus_spec', 'value')])

def update_confirmed_spec(n_clicks, select_focus_spec):

    if n_clicks != 0:    
        if len(select_focus_spec) == 1:
            children = html.P('You have chosen ' +select_focus_spec[0]+ '!')
        else:
            children = html.P('You have chosen ' +", ".join(select_focus_spec)+ '!')
        return children
    else:
        raise PreventUpdate
    

# maximum number of variables that can be the input of our model for each focused spec 
max_var_cnt = 3

# select variable that will be put into model
@app.callback([Output('intermediate_layer3', 'data'),
               Output('intermediate_layer3', 'value')],
              [Input('material_dropdown', 'value'),
                Input('choose_focus_spec', 'value'),
                Input('intermediate_layer', 'value'),
                Input('confirm_spec_button', 'n_clicks')])

def update_input_var(select_material, focus_spec, exist_spec, n_clicks,
                      max_var_cnt = max_var_cnt):
    
    if n_clicks > 0:
        df = spec[spec['material'] == select_material]
        lot_df = function.LOT_based_df(df, exist_spec, 'Original')
        lot_df = lot_df.drop(lot_df.columns[0:2], axis=1)
        
        cor_df = function.corr_var(focus_spec, lot_df)
        cor_df = cor_df.T
        input_var_dic = function.recommend_var(cor_df, max_var_cnt)
        
        with open('input_var_dic.json', 'w') as file:
            json.dump(input_var_dic, file)
        
        cor_df.to_csv('cor_df.csv')
            
        return 'input_var_dic.json', 'cor_df.csv, input_var_dic.json'
    else:
        raise PreventUpdate

# return the dcc.Input for each focus spec
@app.callback([Output('input_space', 'children'),
               Output('button_collection', 'style')],
              [Input('intermediate_layer3', 'data'),
               Input('confirm_spec_button', 'n_clicks')],
              prevent_initial_call = True)

def update_show_input_var(dic_name, n_clicks):
    if n_clicks > 0:            
        with open(dic_name, 'r') as j:
            dic = json.load(j)
            
        key_list = list(dic.keys())
        children = [EDA.create_input_block(dic, key_list[i], i) for i in range(len(key_list))] 
        return children, {'display': 'block'}
    else:
        return PreventUpdate


# pattern-matching callback, obtain the renew_variable and return new suggestion
@app.callback(Output({'type': 'filter-input', 'index': ALL}, 'value'),
              [Input('renew_variable_button', 'n_clicks'),
               Input('intermediate_layer3', 'value')],
              State({'type': 'filter-input', 'index': ALL}, 'value'),
              prevent_initial_call = True)

def renew_show_input_var(renew_click, names, values,
                          max_var_cnt = max_var_cnt):
   
    if renew_click > 0:
        name_list = names.split(', ')
        df = pd.read_csv(name_list[0])
        
        with open(name_list[1], 'r') as j:
            old_dic = json.load(j)
        
        update_dic = {i: value for (i,value) in enumerate(values)}
        new_suggest_dic = function.update_recommend_var(df, update_dic, 
                                                        old_dic, max_var_cnt)
        
        with open('input_var_dic.json', 'w') as file:
            json.dump(new_suggest_dic, file)
        
        key_list = list(new_suggest_dic.keys())
        input_vec = [', '.join(new_suggest_dic[i]) for i in key_list]
        
        return input_vec
    else:
        return PreventUpdate

# warning for freezing renew_input_variable button    
@app.callback(Output('freeze_confirm', 'displayed'),
              Input('input_confirm', 'n_clicks'))
def display_confirm_table(n_click):
    if n_click > 0:
        return True
    else:
        return False

 
@app.callback([Output('input_space', 'style'),
                Output('variable_button', 'style'),
                Output('model_button', 'style'),
                Output('show_freeze_table', 'children')],
              [Input('input_confirm', 'n_clicks'),
                Input('freeze_confirm', 'submit_n_clicks'),
                Input('unfreeze', 'n_clicks'),
                Input('choose_focus_spec', 'value')],
              [State({'type': 'filter-input', 'index': ALL}, 'value')])

def display_and_show_table(confirm_click, freeze_click, unfreeze_click,
                            focus_spec, values):

    if freeze_click is not None and confirm_click > 0:
        if freeze_click - unfreeze_click > 0:
            new_table = EDA.generate_freeze_table(focus_spec, values)
            table = dash_table.DataTable(
                        id='freeze_table',
                        columns=[{"name": i, "id": i} for i in new_table.columns],
                        data=new_table.to_dict('records'),
                    )
        
            return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, table
        else:
            return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True
    else:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True    
    

@app.callback(Output('model_confirm_dialogue', 'displayed'),
              Input('run_model', 'n_clicks'))
def display_confirm_model(model_click):
    if model_click > 0:
        return True
    return False


@app.callback([Output('intermediate_layer4', 'data'),
               Output('model_confirm_dialogue_body', 'children')],
              [Input('run_model', 'n_clicks'),
               Input('material_dropdown', 'value'),
               Input('intermediate_layer3', 'data')],
               prevent_initial_call = True)

def update_stan_name_and_dialogue(model_click, select_material, dic_name):
    
    if model_click > 0:
        with open(dic_name, 'r') as j:
            dic = json.load(j)
            
        spec_last = spec['date'].max()
        deal_last = deal['date'].max()
        info_dic = function.create_info_dic(select_material, spec_last, deal_last, dic)

        next_name_cnt = function.name_next_stan_file(os.getcwd())
        next_name = select_material + '_stan_result' + next_name_cnt + '.json'
        next_info_name = select_material + '_info' + next_name_cnt + '.json'
        
        final_dic = {'next_name': next_name,
                     'next_info_name': next_info_name,
                     'info_dic': info_dic}
        
        children = EDA.create_model_dialogue(info_dic)
        
        return final_dic, children
    else:
        raise PreventUpdate


@app.callback([Output('info_store1', 'data'),
               Output('next_page', 'style')],
              [Input('material_dropdown', 'value'),
               Input('intermediate_layer3', 'data'),
               Input('intermediate_layer4', 'data'),
               Input('run_model', 'n_clicks'),
               Input('model_confirm_dialogue', 'submit_n_clicks')])

def update_run_model(select_material, dic_name, stan_info, model_click, dialogue_click):
    
    if model_click > 0 and dialogue_click is not None:
        print('run model')
        with open(dic_name, 'r') as j:
            dic = json.load(j)
        
        concat_df, mean_var_dic = function.spec_deal_concat(select_material, 
                                                           spec, deal, dic, 'Standardized')

        

        global stan_result
        stan_result, purchase_info_dic = function.run_stan(concat_df, dic)
        
        stan_info['info_dic']['Mean and Var'] = mean_var_dic
        stan_info['info_dic']['purchase_info'] = purchase_info_dic
        
        with open(stan_info['next_name'], 'w') as file:
            json.dump(stan_result, file)
        with open(stan_info['next_info_name'], 'w') as file:
            json.dump(stan_info['info_dic'], file)
        
        return stan_info['info_dic'], {'float': 'right'}
    else:
        raise PreventUpdate
                
        
###########################################################


######## Callback for Manage_Data #########################
@app.callback(Output('info_card', 'children'),
              Input('intermediate_layer5', 'data'))

def update_info_card(select_material):
    
    mypath = './' + select_material
    dic_list = function.find_usable_file(mypath)
    
    if len(dic_list) == 0:
        return Manage_Data.nothing_available(True)
    else:
        children = [Manage_Data.create_info_card(j, i) for i,j in enumerate(dic_list)]
        return children

    
@app.callback([Output({'type': 'dynamic-Card', 'index': MATCH}, 'color')],
              [Input({'type': 'dynamic-choose-button', 'index': MATCH}, 'n_clicks'),
                Input({'type': 'dynamic-delete-button', 'index': MATCH}, 'n_clicks'),
                Input({'type': 'dynamic-recover-button', 'index': MATCH}, 'n_clicks')],
              State({'type': 'dynamic-Card', 'index': MATCH}, 'color'),
              prevent_initial_call = True)


def update_card_color(choose_value, delete_value, recover_value,
                      current_color):
    choose_color = 'red'
    delete_color = 'light'
    recover_color = 'dark'

    if choose_value > 0:
        return [choose_color]
    if (delete_value - recover_value) > 0 and (current_color == recover_color):
        return [delete_color]
    elif (delete_value - recover_value) == 0 and (current_color == delete_color):
        return [recover_color]
    else:
        return [recover_color]
    

@app.callback([Output({'type': 'dynamic-delete-button', 'index': MATCH}, 'style'),
               Output({'type': 'dynamic-recover-button', 'index': MATCH}, 'style')],
              [Input({'type': 'dynamic-delete-button', 'index': MATCH}, 'n_clicks'),
               Input({'type': 'dynamic-recover-button', 'index': MATCH}, 'n_clicks')])

def hide_delete_button(delete_value, recover_value):
    if (delete_value - recover_value) == 0:
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}
    

@app.callback(Output({'type': 'dynamic-choose-button', 'index': ALL}, 'style'),
              Input({'type': 'dynamic-choose-button', 'index': ALL}, 'n_clicks'))

def hide_choose_button(choose_value):
    if sum(choose_value) > 0:
        return [{'display': 'none'} for i in range(len(choose_value))]
    else:
        return [{'display': 'block'} for i in range(len(choose_value))]


@app.callback(Output('confirm_import', 'href'),
              Input({'type': 'dynamic-choose-button', 'index': ALL}, 'n_clicks'))

def renew_import_confirm_href(choose_value):
    if sum(choose_value) == 1:
        return '/Model'
    else:
        return '/Manage_Data'
    
                                             
@app.callback([Output('warning_dialogue', 'message'),
               Output('warning_dialogue', 'displayed'),
               Output('info_store2', 'data')],
              [Input({'type': 'dynamic-choose-button', 'index': ALL}, 'n_clicks'),
                Input({'type': 'dynamic-delete-button', 'index': ALL}, 'n_clicks'),
                Input({'type': 'dynamic-recover-button', 'index': ALL}, 'n_clicks'),
                Input('confirm_import', 'n_clicks'),
                Input('intermediate_layer5', 'data')],
              prevent_initial_call = True)

def execute_and_move(choose_value, delete_value, recover_value, confirm_click,
                      select_material):
    
    if confirm_click > 0:
        mypath = './' + select_material
        os.chdir(mypath)
        
        if sum(choose_value) == 0:
            function.delete_and_rename_file(mypath, delete_value, 
                                        recover_value, choose_value, select_material)
            return Manage_Data.warning_dialogue(0), True, None
        elif sum(choose_value) == 1:
            global stan_result
            
            file_name = select_material + '_stan_result' + str(choose_value.index(1)+1) + '.json'
            info_name = select_material + '_info' + str(choose_value.index(1)+1) + '.json'
            
            with open(file_name, 'r') as j:
                stan_result = json.load(j)
            with open(info_name, 'r') as j:
                info_dic = json.load(j)
                
            function.delete_and_rename_file(mypath, delete_value, 
                                        recover_value, choose_value, select_material)    
            return Manage_Data.warning_dialogue(1), False, info_dic
        else:
            function.delete_and_rename_file(mypath, delete_value, 
                                        recover_value, choose_value, select_material)
            
            return Manage_Data.warning_dialogue(2), True, None
    else:
        raise PreventUpdate
                                                 
        
###########################################################
        

######## Callback for Model ###############################
    
@app.callback(Output('customer_slider', 'value'),
              Input('class_cnt', 'value'))

def update_init_customer_slider(value):
    segment = 1/value
    vec = [i*segment for i in range(1,value)]
    
    return vec

@app.callback([Output('input_spec_inform_div', 'children'),
                Output('focus_spec_intermediate', 'data')],
                [Input('info_store1', 'data'),
                Input('info_store2', 'data'),
                Input('intermediate_layer5', 'data')])

def update_input_df(info1, info2, select_material):
    
    temp_dic = function.return_table_variable(info1, info2, select_material)
    try:
        reserve_spec = temp_dic['input']
    except:
        warning = html.H5('''Spec information hasn't been imported, please check again!''')
        return warning, None
    
    data_dic = {'LOTNO': 'LOTNO'}
    for i in reserve_spec: data_dic[i] = 0
    children = dash_table.DataTable(id='input_spec_inform',
                                    columns=[{'name': 'LOTNO', 'id': 'LOTNO'}] + [
                                        {'name':i, 'id':i} for i in reserve_spec],
                                    data=[data_dic],
                                    editable=True)
    
    return children, temp_dic


@app.callback(Output('rate_and_score_info', 'data'),
              [Input('submit_spec', 'n_clicks'),
                Input('input_spec_inform', 'data'),
                Input('input_spec_inform', 'columns'),
                Input('focus_spec_intermediate', 'data')],
              prevent_initial_call = True)

def update_input_spec(submit_click, row, column, dic):
    
    if submit_click > 0 and isinstance(stan_result, dict):
        focus_spec = dic['focus']
        input_spec = dic['input']
        input_spec_value = [float(row[0][i]) for i in input_spec]
        
        rate, score = function.rating_system(input_spec_value, focus_spec,
                                              stan_result, input_spec, dic['Mean and Var'])
        
        return {'rate': rate.to_dict(), 'score': score.to_dict()}
    else:
        raise PreventUpdate

@app.callback([Output('recommendation_system_table', 'children'),
                Output('recommendation_system_plot', 'children'),
                Output('total_ranking_store', 'data')],
              [Input('submit_spec', 'n_clicks'),
                Input('customer_slider', 'value'),
                Input('focus_spec_intermediate', 'data'),
                Input('rate_and_score_info', 'data'),
                Input('intermediate_layer5', 'data')])

def update_recommendation_system(submit_click, cut_off, info_dic,
                                 score_dic, select_material):

    if submit_click > 0:
        for i in score_dic.keys():
            score_dic[i] = pd.DataFrame(score_dic[i])
        cut_off = [round(i,2) for i in cut_off]
            
        choose_cnt = 5
        
        cust_dic, quantile_record = function.cust_to_class(cut_off, info_dic['purchase_info'])
        
        
        rating_class_dic, class_score = function.cust_rating(cust_dic, score_dic['score'],
                                                choose_cnt, False, customer)
        ranking_table = function.score_name_congugate(rating_class_dic, class_score)
        table_children = Model.dic_to_table(ranking_table, choose_cnt)
        
        rating_all_class_dic, all_class_score = function.cust_rating(cust_dic, score_dic['score'],
                                                      'total', False, customer)
        rating_all_dic, all_score = function.cust_rating(cust_dic, score_dic['score'],
                                              'total', True, customer)
        figure_children = Model.create_figure(rating_all_class_dic,  all_class_score,
                                              rating_all_dic, all_score,
                                              score_dic, customer,
                                              info_dic['purchase_info'],
                                              select_material,
                                              choose_cnt)
        
        store_dic = {'by_class': rating_all_class_dic, 'all': rating_all_dic}
        
        return table_children, figure_children, store_dic
    else:
        raise PreventUpdate
        
@app.callback(Output({'type': 'dynamic_plot', 'index': ALL}, 'style'),
              [Input('class_plot', 'value'),
               Input('class_plot', 'options')])

def show_final_class_plot(class_name, option):
    class_cnt = len(option)
    option_list = [option[i]['label'] for i in range(class_cnt)] 
    
    index = option_list.index(class_name)
    style_list = [{'display': 'none'} for i in range(class_cnt)]
    style_list[index] = {'display': 'block'}
    
    return style_list
    

    
        

###########################################################

# needed only if running this as a single page app
server = app.server

if __name__ == '__main__':
    app.run_server(debug = False, 
                   port = 9487) 



