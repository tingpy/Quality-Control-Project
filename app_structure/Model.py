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
        rate.sort_values(by='score', ascending=True)
        rate['rank'] = [i+1 for i in range(rate.shape[0])]
            
        if len(spec_name) == 1:
            mean_vec = list(rate[spec_name[0] + '_mean'])[0: choose_cnt]
            sd_vec = list(rate[spec_name[0] + '_sd'])[0: choose_cnt]
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
    rate['score'] = all_score['ALL']
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


def update_data_conditional_style(cur_data):
    return {}
# def discrete_background_color_bins(dic, n_bins, columns='all'):
#     import colorlover
#     ranges = [-10000, -4, -3, -2, 2, 3, 4, 10000]
#     background = colorlover.scales['3']['seq']['Reds'].reverse() + [''] + colorlover.scales['3']['seq']['Reds']
#     styles = []
#     legend = []
#     df = pd.DataFrame(dic)
#     col_name = df.columns
    
#     for i in 
#     for i in range(1, len(bounds)):
#         min_bound = ranges[i - 1]
#         max_bound = ranges[i]
#         backgroundColor = background[i]

#         for column in df.columns:
#             styles.append({
#                 'if': {
#                     'filter_query': (
#                         '{{{column}}} >= {min_bound}' +
#                         (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
#                     ).format(column=column, min_bound=min_bound, max_bound=max_bound),
#                     'column_id': column
#                 },
#                 'backgroundColor': backgroundColor,
#                 'color': color
#             })
#         legend.append(
#             html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
#                 html.Div(
#                     style={
#                         'backgroundColor': backgroundColor,
#                         'borderLeft': '1px rgb(50, 50, 50) solid',
#                         'height': '10px'
#                     }
#                 ),
#                 html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
#             ])
#         )

#     return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))

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
                                               n_clicks=0))
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
                                   html.Div(id='recommendation_system_plot'),
                                   
                                   ])
                               ]),
                           
            ])