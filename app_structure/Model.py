# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:13:54 2021

@author: acer
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

# import plotly.figure_factory as ff
from plotly.figure_factory import create_distplot
from plotly.graph_objects import Figure, Scatter

import numpy as np
import pandas as pd


def dic_to_table(dic, choose_cnt):
    column = ['Class'] + [str(i+1) for i in range(choose_cnt)]
    
    data = []
    for i in dic.keys():
        new_dic = {'Class': i}
        for j,k in enumerate(dic[i]):
            new_dic[column[j+1]] =  k
        data.append(new_dic)
    
    column_dic = [{'name': i, 'id': i} for i in column]
    
    table = dash_table.DataTable(
                id='customer_list',
                columns=column_dic,
                data=data,
            )
    
    return table

def figure_generate(dic, rate, select_material, choose_cnt):
    # check how many focus_spec is there
    sub = '_mean'
    spec_name = [i.split(sub)[0] for i in rate.columns if sub in i]
    
    figure_children = []
    rate_df = rate
    multiple_bool = False
    for i in dic.keys():
        if i != 'ALL':
            # filter rate, preserve customers in current class
            multiple_bool = True
            rate = rate_df
            rate = rate.loc[rate['class'] == i, [True for i in range(rate.shape[1])]]
        # append column 'rank'
        rate = rate.sort_values(by='score', ascending=True)
        rate = rate.reset_index(drop = True)
        rate['rank'] = [i+1 for i in range(rate.shape[0])]
        
            
        if len(spec_name) == 1:
            mean_vec = list(rate[spec_name[0] + '_mean'])[0: choose_cnt]
            sd_vec = list(rate[spec_name[0] + '_sigma'])[0: choose_cnt]
            data = [mean_vec[i] + sd_vec[i]*np.random.randn(1000) 
                    for i in range(choose_cnt)]
            label = list(rate['real name'])[0: choose_cnt]
            
            graph = create_distplot(data, 
                                    label, 
                                    show_hist=False)
            graph.update_layout(title_text = select_material+' focus on '+spec_name[0],
                                xaxis_title = spec_name[0] + sub,
                                yaxis_title = 'Probability')  
            
        elif len(spec_name) == 2:
            x = list(rate[spec_name[0]+sub])
            y = list(rate[spec_name[1]+sub])
            text = []
            for index, row in rate.iterrows():
                text.append(('Company Name: {name}<br>'+
                             'Class: {class_level}<br>'+
                             'Count of total purchase: {purchase_cnt}<br>'+
                             spec_name[0] + ': {spec1}<br>'+
                             spec_name[1] + ': {spec2}<br>'+
                             'Score: {score}<br>'+
                             'Rank: {rank}<br>').format(name=row['real name'],
                                                        class_level=row['class'],
                                                        purchase_cnt=row['purchase cnt'],
                                                        spec1=round(row[spec_name[0]+'_mean'],3),
                                                        spec2=round(row[spec_name[1]+'_mean'],3),
                                                        score=round(row['score'],3),
                                                        rank=row['rank']))
            size = [i**0.5 for i in rate['purchase cnt']]
            color = rate['rank']
            graph = Figure(data=[Scatter(
                            x = x, y = y,
                            text = text,
                            mode='markers',
                            marker=dict(size = size,
                                        color = color,
                                        showscale=True,
                                        sizemode='area',
                                        sizeref=2.*max(size)/(40.**2),
                                        sizemin=4)
                        )])
            # graph = go.Scatter(rate,
            #                     x = spec_name[0]+sub,
            #                     y = spec_name[1]+sub,
            #                     text = text,
            #                     color = "rank",
            #                     mode='markers',
            #                     marker=dict(size = size,
            #                                 sizemode='area',
            #                                 sizeref=2.*max(size)/(40.**2),
            #                                 sizemin=4))
            
            graph.update_layout(title_text = select_material + ' focus on ' + spec_name[0] + \
                                ' and ' + spec_name[1] + '  (For ' + i + ' customer)',
                                xaxis_title = spec_name[0] + sub,
                                yaxis_title = spec_name[1] + sub)
        
        if multiple_bool:
            figure_children.append(dbc.Row([dcc.Graph(id={'type': 'dynamic_plot',
                                                          'index': i},
                                                      figure=graph,
                                                      style={'display': 'none'})]))
        else:
            figure_children.append(dbc.Row([dcc.Graph(figure=graph)]))
        
    return figure_children
    


def create_figure(class_dic, class_score, all_dic, all_score, score_dic,
                  customer, purchase_info, select_material, choose_cnt):
    rate = score_dic['rate']
    all_no = [str(j) for j in customer['客戶編號']]
    all_name = list(customer['名稱'])
    
    real_name = []
    for i in rate['customer id']:
        try:
            index = all_no.index(i)
            real_name.append(all_name[index])
        except:
            real_name.append(i)
    rate['real name'] = real_name
    rate = pd.merge(rate, score_dic['score'], on='customer id')
    rate['purchase cnt'] = purchase_info['purchase_cnt']
    
    # add customer class to dataframe 'rate'
    class_reserve = [0 for i in range(rate.shape[0])]
    exist_name = list(rate['real name'])
    for i in class_dic.keys():
        for j in class_dic[i]:
            index = exist_name.index(j)    
            class_reserve[index] = i
    rate['class'] = class_reserve
    
    if len(list(class_dic.keys())) == 1:
        all_figure = figure_generate(all_dic, rate, select_material, choose_cnt)
        
        return all_figure
    else:
        all_figure = figure_generate(all_dic, rate, select_material, choose_cnt)
        seperate = figure_generate(class_dic, rate, select_material, choose_cnt)
        
        class_dropdown = [dbc.Row([
            dcc.Dropdown(id='class_plot',
                         options=[{'label': i, 'value':i}
                                  for i in class_dic.keys()],
                         value=list(class_dic.keys())[0])
        ])]
        
        card = [dbc.Row([
                dbc.Col(
                    dbc.Card(
                        html.H3(children='Choose the class of customer you want to inspect',
                                className="text-center text-light bg-dark"), 
                    body=True, color="dark"),
                    className="mb-4")
                ])]
        
        sub_children = html.Div(
            all_figure + card + class_dropdown + seperate
        )
        
        return sub_children  
    

def generate_history_row(customer_id, cust_history, input_val,
                         col_name, rank, cust_name):
    row = [cust_name, rank+1]
    
    df = cust_history[cust_history['buyer'] == customer_id]
    
    digit = 4
    focus = input_val['spec name']
    mean = input_val['mean']
    sd = input_val['sd']
    mean_val = [round(np.mean(df[j]) * sd[i] + mean[i], digit) for i, j in enumerate(focus)]
    
    value = input_val['value']
    prob = []
    total_bool = [False for i in range(0, df.shape[0])]
    for i, j in enumerate(focus):
        standardize = (value[i] - mean[i]) / sd[i]
        if standardize < 0:
            temp_bool = df[j] <= standardize
            prob.append(round(np.mean(temp_bool), digit))
        else:
            temp_bool = df[j] >= standardize
            prob.append(round(np.mean(temp_bool), digit))
        total_bool = np.logical_or(total_bool, temp_bool)
            
    if len(focus) > 1:
        prob = prob + [round(np.mean(total_bool), digit)]
    
    row = row + mean_val + prob
    dic = {}
    for i, j in enumerate(row):
        dic[col_name[i]] = j
        
    return dic
    



layout = dbc.Container(id='model_body',
                       children=[
                           dbc.Row([
                              dbc.Col(
                                  dbc.Card(
                                      html.H3(children='Customer Segmentation',
                                              className="text-center text-light bg-dark"), 
                                  body=True, color="dark"),
                                  className="mb-4")
                              ]),
                           
                           # dbc.Row([
                           #     html.H4('Choose the product level you want to fucus')
                           #     ]),
                           
                           # dbc.Row([
                           #     dcc.Dropdown(id='level_dropdown',
                           #                  options=[{'label': i, 'value': i}
                           #                           for i in ['first level', 'secondary','both']],
                           #                  value='first level')
                           #     ]),
                           
                           dbc.Row([
                               html.H4('Choose the number of class and quantile for each class')
                               ]),
                           
                           dbc.Row([
                               dbc.Col([
                                   dcc.Dropdown(id='class_cnt',
                                                options=[{'label': i, 'value':i}
                                                         for i in range(3,7)],
                                                value=3)
                                   ])
                               ]),
                           
                           dbc.Row([
                               dbc.Col([
                                   dcc.RangeSlider(id='customer_slider',
                                              min=0, max=1,
                                              step=None,
                                              marks={i: str(int(i*100))+'%' for i in 
                                              np.linspace(start=0, stop=100,num=21)/100})
                                   ])
                               ]),
                               # show the table about customer_cnt
                               # dbc.Col()
                               
                           html.Br(),
                           
                           dbc.Row([
                              dbc.Col(
                                  dbc.Card(
                                      html.H3(children='Ranking Setting',
                                              className="text-center text-light bg-dark"), 
                                  body=True, color="dark"),
                                  className="mb-4")
                              ]),
                           
                           dbc.Row([
                               dbc.Col(html.H5('Choose the ranking method you want to apply.'))
                               ]),
                           
                           dbc.Row([
                               dbc.Col(
                                   dcc.Dropdown(id='ranking_method',
                                                options=[{'label': 'Method ' + str(i), 'value': i}
                                                         for i in [1, 2]],
                                                value=1))
                               ]),
                           
                           html.Br(),
                           
                           dbc.Row([
                               dbc.Col(
                                   html.Div(id='ranking_method_intro')
                                   )
                               ]),
                           
                           html.Br(),
                           
                           dbc.Row([
                               dbc.Col(html.H5('Choose number of customer you want to show in the table.'))
                               ]),
                            
                           dbc.Row([
                               dbc.Col([
                                   dcc.Dropdown(id='ranking_cnt',
                                                value=5)
                                   ])
                               ]),
                           
                           html.Br(),
                              
                           dbc.Row([
                              dbc.Col(
                                  dbc.Card(
                                      html.H3(children='Input The Spec Information',
                                              className="text-center text-light bg-dark"), 
                                  body=True, color="dark"),
                                  className="mb-4")
                              ]),
                           
                           dbc.Row([
                               dbc.Col(html.Div(id='input_spec_inform_div'))
                              ]),
                           
                           dbc.Row([
                               dbc.Col(
                                   html.Button('Submit Spec', 
                                               id='submit_spec',
                                               n_clicks=1))
                              ]),
                           
                           html.Br(),
                           
                           dbc.Row([
                              dbc.Col(
                                  dbc.Card(
                                      html.H3('Customer Recommendation',
                                              className="text-center text-light bg-dark"), 
                                  body=True, color="dark"),
                                  className="mb-4")
                              ]),
                           
                           dbc.Row([
                               dbc.Col([
                                   # html.Div([
                                   #     dcc.Store(id='focus_spec_intermediate'),
                                   #     dcc.Store(id='rate_and_score_info')
                                   #     ]),
                                   html.Div(id="recommendation_system_table"),
                                   
                                   html.Br(),
                                   
                                   html.Div(id='history_category_div',
                                            children = [
                                                dbc.Row(
                                                    dbc.Col(
                                                        dbc.Card(
                                                            html.H3(children='History Countercheck',
                                                                    className="text-center text-light bg-dark"), 
                                                        body=True, color="dark")
                                                        )
                                                    ),
                                                
                                                html.Br(),
                                                
                                                dbc.Row(
                                                    dbc.Col([
                                                        html.Div('Choose the class of customers you want to inspect'),
                                                        dcc.Dropdown(id='history_plot_category'),
                            
                                                        html.Br(),
                                                        ])
                                                    ),
                                                
                                                dcc.ConfirmDialog(
                                                        id='append_cust_warning',
                                                    ),
                                                        
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Div('Choose the customer you want to inspect'),
                                                        html.Div(id='customer_in_category_dropdown_div')
                                                       ]),
                                                   
                                                   dbc.Col(
                                                       dbc.Row([
                                                           html.Div('Key in the rank of the customer you want inspect'),
                                                           dcc.Input(id='customer_in_category_rank'),
                                                           html.Div(id='max_rank_notification')]),
                                                       ),
                                               
                                                   dbc.Col(html.Button('Append New Customer',
                                                                      id='customer_in_category_button',
                                                                      n_clicks=0)
                                                       )
                                                    ]),
                                                
                                                dbc.Row(
                                                    dbc.Col([
                                                        html.H4('history table'),
                                                        html.Div(id='history_stat_table_div'),
                                                        ])
                                                    ),
                                                
                                               ], style = {'display': 'none'}),
                                   
                                   html.Div(id='recommendation_system_plot'),
                                   
                                   ])
                               ]),
                           
            ])