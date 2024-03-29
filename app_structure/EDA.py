# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:13:32 2021

@author: acer
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
# import plotly.figure_factory as ff
from plotly.figure_factory import create_distplot
# import plotly.express as px
from plotly.express import scatter
# from plotly.offline import plot

import pandas as pd
from datetime import datetime

from app_structure import function
# from quality_app import deal
# from quality_app import spec

# still have some problem in the plot
def create_spec_plot(df, select_spec, table_type, data_type):
    if not isinstance(select_spec, list):
        select_spec = [select_spec]
        
    if len(select_spec) == 0:
        return []
    
    if table_type == 'Density Plot':
        plot_dic = function.spec_to_dic(df, select_spec, data_type)
        hist_data = [plot_dic[i] for i in plot_dic.keys()]
        group_label = [i for i in plot_dic.keys()]
        
        return create_distplot(hist_data, group_label, bin_size=.2)
    elif table_type == 'Time Shift Plot':
        if len(select_spec) == 1:
            plot_dic = function.time_shift(df, select_spec, data_type)
            hist_data = [plot_dic[i] for i in plot_dic.keys()]
            group_label = [i for i in plot_dic.keys()]
            
            return create_distplot(hist_data, group_label, bin_size=.2)
        else:
            return []
    else:
        if len(select_spec) == 2:
            plot_df = function.LOT_based_df(df, select_spec, data_type, False)
        
            return scatter(plot_df, 
                              x=plot_df.columns[2], # remove 'LOTNO' and 'date'
                              y=plot_df.columns[3])
        else:
            return []


def create_focus_spec_dropdown(material, option):

    children = [
        dbc.Row([
            dcc.Dropdown(
                id='choose_focus_spec',
                multi=True,
                value=['色相(調) - L', 'Film - 薄片污點(2)'],
                options=option,
                persistence=material,
                persistence_type='local',
                style={'width': '100%', 'height': '40px'})
        ])
    ]
    
    return children
            
def create_input_block(dic, key, key_cnt):
    return dbc.Row([
                dbc.Col(html.H3(key)),
                
                dbc.Col(dcc.Input(id={
                                    'type': 'filter-input',
                                    'index': key_cnt,
                                  },
                                  value= ", ".join(dic[key]),
                                  style={'width': '100%'}))
                ])

def generate_freeze_table(focus_spec, select_dic):
    new_dic = {focus_spec[i]: select_dic[i].split(', ') for i in range(len(focus_spec))}
    new_df = pd.DataFrame(new_dic).T
    new_df.columns = ['Feature '+ str(i+1) for i in range(new_df.shape[1])]
    new_df.insert(loc=0, column='Chosen Spec', value=focus_spec)
    
    return new_df

def create_model_dialogue(dic):
    
    confirm_str_warning1 = 'Check the below data information:\n\n' 
    confirm_str_info = '\n'.join([i +':\n\t\t'+ dic[i] for i in dic.keys()])
    confirm_str_warning2 = '''\n\nRun model will take some time, please wait patiently!
    Click to continue'''
    
    return dbc.Row([dcc.ConfirmDialog(
        id='model_confirm_dialogue',
        message=confirm_str_warning1 + confirm_str_info + confirm_str_warning2)
    ])
    
def generate_duration_table(df, select_spec):
    children = []
    
    for i in select_spec:
        temp_df = df[df['spec name'] == i]
        unique_limit = set(temp_df['spec limit'])
        store_df = pd.DataFrame(data = None,
                                columns = ['First Date', 'Last Date', 'Limit', 'Selected'])

        for j, k in enumerate(unique_limit):
            limit_df = temp_df[temp_df['spec limit'] == k]
            max_date = limit_df['date'].max()
            min_date = limit_df['date'].min()
            max_date = str(max_date.year) + '/' + str(max_date.month)
            min_date = str(min_date.year) + '/' + str(min_date.month)
            
            print([min_date, max_date, k, ''])
            store_df.loc[j] = [min_date, max_date, k, 'Unselected']
        
        store_df = store_df.sort_values(by=['Last Date'], ascending=False)
        
        children.append(
            html.Div([
                dbc.Row(
                    dbc.Col(html.H5(i + ':'))),
                
                dbc.Row(
                    dbc.Col(
                        dash_table.DataTable(
                            id = {'type': 'limit-table',
                                  'index': i},
                            columns = [{'name':z, 'id':z} for z in store_df.columns],
                            data = store_df.to_dict('records'),
                            is_focused = True
                        )
                    )
                ),
                
                html.Br(),
            ])
        )
        
    return children

def generate_time_slider(now):
    marks_dic = {}
    for i in range(now.year, datetime.now().year+1):
        marks_dic[datetime.timestamp(datetime.strptime(str(i)+'/1/1',
                                                       '%Y/%m/%d'))] = {'label': str(i)}
        
    slider = dcc.RangeSlider(
        id = 'material_slider',
        min = datetime.timestamp(now),
        max = datetime.timestamp(datetime.now()),
        value = [datetime.timestamp(datetime.strptime('2019/1/1','%Y/%m/%d')),
                 datetime.timestamp(datetime.now())],
        marks = marks_dic,
        persistence_type = 'local'
    )
    
    return slider

def generate_line_dropdown(fac_inform):
    row_component = []
    for j in fac_inform['line'].keys():
        print
        tmp_line = fac_inform['line'][j]
        tmp_dropdown = dcc.Dropdown(
                              id={
                                'type': 'dynamic-dropdown',
                                'index': j,
                              },
                              options=[{'label':k, 'value':k}
                                       for k in tmp_line],
                              value=tmp_line,
                              multi=True)
        row_component.append(
            dbc.Row([
                dbc.Col([html.H5(j)],
                        style={'width':'10rem'}),
                dbc.Col([tmp_dropdown],
                        style={'width':'20rem'}),
                ])
            )
    
    return row_component

def generate_select_spec_div(select_spec, select_limit):
    new_div = html.div(
        id={
          'type': 'dynamic-div',
          'index': select_spec,
        },
        children=[html.H5()])
    


material_selection = dbc.Row([
    dbc.Col(
        dbc.Card(
            html.H3(
                children='Please Select The Material You Want To Inspect',
                className="text-center text-light bg-dark"
            ), 
            body=True,
            color="dark"
        ),
        className="mb-4"
    )
])

material_dropdown =html.Div([ 
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='material_dropdown',
                value='765AXXX',
                persistence=True,
                persistence_type='session',
                style={'background-color': 'white','color' : 'black' ,
                       'width': '100%','font-weight': '1000'},
            ),
            className='mb-4')
    ]),
    
    dbc.Row([
        dbc.Col(
            html.Button(
                children=html.A(
                    children='Skip Model',
                    href='/Manage_Data',
                    style={'display':'inline-block', 'width': '200px', 'margin':'auto'}
                    # style={'width': '200px', 'height': '50px','margin-right': '7px',
                    #        'backgroundColor': 'grey', 'color':'white'}
                ),
                id='skip_model',
                n_clicks=0),
            align='end'
        ),
    ]),
])

period_selection = dbc.Row([
    dbc.Col(
        dbc.Card(
            html.H3(
                children='Please Select The Period You Want To Inspect',
                className="text-center text-light bg-dark"
            ), 
            body=True,
            color="dark"
        ),
        className="mb-4"
    )
])

material_slider = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(id='show_slider_date'),
        ]),
        
        dbc.Col([
            html.Button('Change first date',
                        id='min_date_confirm',
                        n_clicks=0),
            ]),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='change_date_div',
                     children=[
                         dbc.Row([
                             dbc.Col([
                                 html.H4('Enter the year you want to start with:')
                             ]),
                             
                             dbc.Col([
                                 dcc.Input(
                                     id='input_the_min_date',
                                     value=2014
                                 ),
                             ]),
                         ]),
                     ],
                     style={'display': 'none'})
            ])
        ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='material_slider_div',
                     children=[
                         generate_time_slider(datetime.strptime('2011/1/1', '%Y/%m/%d'))
                     ]),
        ])
    ])
])

## (spec_inform can be refined)
spec_inform = dbc.Row([
    html.Div([
        dcc.Tabs(id='spec_inform_and_visual',
                 value='spec_1',
                 children=[
                     dcc.Tab(id='info_tab',
                             label='SPEC information',
                             value='spec_1',
                             children=html.Div([
                                 dbc.Col(html.Div(id='spec_inform_update')),
                             ])
                     ),
                     dcc.Tab(id='plot_tab',
                             label='Factory information',
                             value='spec_2',
                             children=html.Div([
                                 # dbc.Row([
                                 #     dbc.Col([
                                 #         html.H5('How many SPECs do you want to select?'),
                                 #         ]),
                                     
                                 #      dbc.Col([
                                 #          dcc.Dropdown(id='spec_plot_cnt',
                                 #                       options=[{'label':j, 'value':j} 
                                 #                                for j in range(1,4)],
                                 #                       value=1),
                                 #          ]),
                                 #      ]),
                                 
                                 dbc.Row([
                                     dbc.Col([
                                         dbc.Row([
                                             html.Br(),
                                             
                                             dbc.Col([
                                                 html.H5('Factory'),
                                             ]),
                                                
                                             dbc.Col([
                                                 dbc.Row([
                                                     dcc.Dropdown(id='factory_inform_dropdown',
                                                                  multi=True)
                                                 ]),
                                             ]),
                                         ]),
                                             
                                        dbc.Row([
                                                 html.Div(id='select_line_div',
                                                          style={"overflow": "scroll",
                                                                 "height": "15rem"}),
                                                 
                                                 dbc.Row([
                                                     dbc.Col([
                                                         html.Br(),
                                                         
                                                         html.H5('QC List'),
                                                         
                                                         html.Div(
                                                             children=[
                                                                 dcc.RadioItems(id='QC_inform_radio',
                                                                                labelStyle={'display': 'block'})],
                                                             style={
                                                                "overflow": "scroll",
                                                                "width": "18rem",
                                                                "height": "10rem",
                                                                # "padding": "2rem 1rem",
                                                                "background-color": "#f8f9fa"
                                                             }),
                                                      ]),
                                                     
                                                     dbc.Col([
                                                         html.Br(),
                                                         
                                                         html.H5('QC Limit'),
                                                         
                                                         html.Div(
                                                             children=[
                                                                 dcc.RadioItems(id='QC_limit_radio',
                                                                                labelStyle={'display': 'block'})],
                                                             style={
                                                                "overflow": "scroll",
                                                                "width": "18rem",
                                                                "height": "10rem",
                                                                # "padding": "2rem 1rem",
                                                                "background-color": "#f8f9fa"
                                                             }),
                                                      ]),
                                                 ]),
                                            ]),
                                        ]),
                                             
                                    dbc.Col([
                                        html.Div(id='factory_line_table'),
                                        
                                        # html.Div(id='multi_select_info_div',
                                        #          children=[
                                        #              html.Div(id='multi_select_info_in',
                                        #                       children=[]),
                                                     
                                        #              dbc.Row([
                                        #                  html.Button('Append New SPEC',
                                        #                          id='multi_select_confirm_button',
                                        #                          n_clicks=0),
                                        #                  ]),
                                        #              ]),
                                    ]),
                                ]),
                                     
                                     
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id='factory_line_graph')
                                    ]),
                                ]),
                           ])       
                     ),
                 ])
        ]),

    
    dbc.Row(id='spec_EDA_layer', children=[
        dbc.Col(
            html.Div(
                id='spec_EDA', 
                children = [
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Card(
                            html.H3(
                                children='Key In The Spec You Want To Visualize',
                                className="text-center text-light bg-dark"), 
                            body=True,
                            color="dark"),
                    ]),
                    
                    html.Br(),
    
                    dbc.Row([
                        dcc.Dropdown(
                            id='select_spec_EDA',
                            multi=True,
                            style={'background-color': 'white','color' : 'black' ,
                                   'width': '100%','font-weight': '1000'}
                        )
                    ]),
                                    
                    html.Br(),
                                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                html.H5(
                                    children='Present Way',
                                    className="text-center text-light bg-dark"), 
                                    body=True,
                                    color="dark"
                                ),
    
                            html.Br(),
    
                            dcc.RadioItems(
                                id='table_type',
                                options=[{'label': i, 'value': i} for i in ['Density Plot', 'Scatter Plot', 'Time Shift Plot']],
                                value='Density Plot',
                                style={"padding": "10px", "max-width": "800px", "margin": "auto",
                                       "flex": "1"},
                                labelStyle={'display': 'inline-block','margin-right': '7px',
                                            'font-weight': 300})
                        ]),
                                    
                        dbc.Col([
                            dbc.Card(
                                html.H5(
                                    children='Preprocessing Way',
                                    className="text-center text-light bg-dark"), 
                                    body=True, 
                                    color="dark"),
                            
                            html.Br(),
                            
                            dcc.RadioItems(
                                id='data_type',
                                options=[{'label': i, 'value': i} for i in ['Original', 'Standardized']],
                                value='Original',
                                style={"padding": "10px", "max-width": "800px", "margin": "auto",
                                       "flex": "1"},
                                labelStyle={'display': 'inline-block','margin-right': '7px',
                                            'font-weight': 300}
                            )
                        ]),
                    ]),
                                    
                    dbc.Row([dcc.Graph(id='spec_EDA_graph',
                                       style={'width': '180vh', 'height': '90vh'})])
                ]
            )
        )
    ]),
])

spec_focus_selection = dbc.Row([
    dbc.Col(
        dbc.Card(
            html.H3(
                children='Please Select The Spec You Want To Focus',
                className="text-center text-light bg-dark"
            ), 
            body=True,
            color="dark"
        ),
    className="mb-4")
])

spec_focus_and_confirm = html.Div([
    dbc.Row([
        dbc.Col([              
            html.Div(id='focus_spec_var',
                     style={ 'background-color': 'white','color' : 'black' ,
                                'width': '150%','font-weight': '1000','float':'left'})
        ]),
    
        dbc.Col([
            html.Button(
                children='Confirmed',
                id='confirm_spec_button', 
                n_clicks=0,
                style={
                    'backgroundColor': 'grey',
                    'color':'white',
                    'width':'50%',
                    'border':'1.5px black solid', 
                    'height': '40px',
                    'text-align':'center', 
                    'marginLeft': '20px',
                    'float': 'right'
                }
            ),
            
            dcc.ConfirmDialog(
                id='duplicated_LOTNO_warning'
            ),
        ]),
    ]),
    
])
                
choose_spec_intermediate = dbc.Row([
    dbc.Col([
        html.Div(id='choose_spec_intermediate')
    ])
])


intermediate_layers = dbc.Row([
    
    dbc.Col([
        html.Div(
            id='input_spec_var',
            children=dbc.Row(
                dbc.Col(
                    html.Div(id='suggest_variable_cnt',
                             style={'background-color': 'white','color' : 'black',
                                    'width': '100%','font-weight': '1000'})
                )
            ),
        ),            
        
        html.Div(id='duration_table'),
        dcc.ConfirmDialog(
            id='duration_unselect',
            message='''You haven't selected the spec limit for all of your focused spec!
            Please check again!''',
        ),
        dcc.ConfirmDialog(
            id='duration_collide',
            message='''Your selection is invalid, no data is included in this duration!
            Please check again!''',
        ),
        
        html.Br(),
        
        html.Div(id='show_duration_info'),
        
        html.Br(),
        
        html.Div(id='input_space',
                 style={'height' : '100%',
                                    'width': '100%','font-weight': '1000','padding': '30px'}),
        html.Div(id='show_freeze_table',style={'height' : '100%','padding': '30px'})
    ])
])


button_collections = html.Div(id='button_collection',
                              children = [
    dbc.Row([
        dbc.Col(
            children=[
                html.Div(
                    dcc.ConfirmDialog(
                        id='freeze_confirm',
                        message='''After this step the variable table will be freezed. 
                        Are you sure you want to continue?''',
                    )
                ),
                html.Div(id='model_confirm_dialogue_body'),
                html.Div(id='skip_model_dialogue_body'), 
                
                
                html.Div(id='All_of_the_buttons',
                         children=[
                             html.Div(
                                id='variable_button',
                                children=[
                                    dbc.Row([
                                        dbc.Col(
                                            html.Button(
                                                children='Renew The Input Variable',
                                                id='renew_variable_button',
                                                n_clicks=0,
                                                style={'width': '200px', 'height': '50px','margin-right': '7px',
                                                       'backgroundColor': 'grey', 'color':'white'})
                                            ),
                                        dbc.Col(
                                            html.Button(
                                            children='Input Confirmed',
                                            id='input_confirm',
                                            n_clicks=0,
                                            style={'width': '200px', 'height': '50px', 'margin-right': '7px',
                                                   'backgroundColor': 'grey', 'color':'white' , 'float':'right'}),
                                        ),
                                    ], align = 'end') 
                                ]
                            ),            
                            
                            html.Div(
                                id='model_button', 
                                children=[
                                    dbc.Row([
                                        dbc.Col(
                                            html.Button(
                                                children='Unfreeze Table',
                                                id='unfreeze',
                                                n_clicks=0,
                                                style={'width': '200px', 'height': '50px', 'margin-right': '7px',
                                                       'backgroundColor': 'grey', 'color':'white'}
                                            ),
                                        ),
                                        dbc.Col(
                                            html.Button(
                                                children='Run Model',
                                                id='run_model',
                                                n_clicks=0,
                                                style={'width': '200px', 'height': '50px','margin-right': '7px',
                                                       'backgroundColor': 'grey', 'color':'white'}
                                            ),
                                        )
                                    ])
                                ],style={'display': 'none'}
                                ),
                    ])
               
            ],
            align='start'
        ),
        dbc.Col(
            html.Button(
                children=html.A(
                    children='Next',
                    href='/Model',
                    style={'width': '200px', 'height': '50px','margin-right': '7px',
                           'backgroundColor': 'grey', 'color':'white'}
                ),
                id='next_page',
                style={'display': 'none'}    
            ),
            align='end'
        )
    ])
], style ={'display': 'none'})


# Layout Structure

layout = dbc.Container(
    id='EDA_body',
    children = [
        material_selection,
        material_dropdown,
        period_selection,
        material_slider,
        spec_inform,
        spec_focus_selection,
        spec_focus_and_confirm,
        choose_spec_intermediate,      
        dbc.Row(
            html.P('We strongly recommend more than 2 and less than 4 variable for each focus spec!')
        ),
        intermediate_layers,
        dbc.Row([
            dbc.Col(
                html.Div(id='try_layer')
            ),
        ]),
        button_collections
    ]   
)