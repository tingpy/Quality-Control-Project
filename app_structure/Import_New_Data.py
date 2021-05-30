# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:13:03 2021

@author: acer
"""

import base64
import datetime
import io
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

import pandas as pd

# import all pages in the app folder
from quality_py import deal
from quality_py import spec
from quality_py import customer
from quality_py import agent


def parse_contents(contents, filename, date, original_df, df_name):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    print("use parse content")
    try:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
        
        if isinstance(original_df, pd.DataFrame):
            try:
                df.columns = original_df.columns
                globals()[df_name] = pd.concat([original_df, df], axis=0)
            except Exception as e:
                print(e)
                return html.Div([
                    html.H4('Warning:'),
                    html.Div([
                    '''The format of the new imported dataframe is wrong. Please check again!''']),
                    dash_table.DataTable(
                        data=original_df.tail(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in original_df.columns]),
                    ])

        
    except Exception as e:
        print(e)
        return html.Div([
                    html.H4('Warning:'),
                    html.Div([
                    '''There was an error processing this file! Please reload your data again.''']),
                    dash_table.DataTable(
                        data=original_df.tail(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in original_df.columns]),
                    ])
        
    return html.Div([
        html.H5(filename),
        html.H6('Upload Time:' + 
                datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")),
    
        dash_table.DataTable(
            data=df.tail(10).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
    
        html.Hr(),  # horizontal line
    
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

# use for show our original data in Import_New_Data
def original_df(which_tab, df):
    warning_str = '''You haven't upload any data to ''' + which_tab + ' before!'
    children = [html.H4(warning_str)]
    
    if isinstance(df, pd.DataFrame):
       children = [dash_table.DataTable(
                            data=df.tail(10).to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in df.columns]),
                       html.Hr(),] 

    return children

# tab content for tabs
def tab_content(which_tab, df):
    print('################')
    print(which_tab)
    
    children=[dbc.Col([
                 dbc.Row([
                 html.H4('The example csv format:')
                 ]),
                 dbc.Row([
                     ### add image here ###
                     #html.Img()
                     ]),
                 dbc.Row([
                     dcc.Upload(id='upload_data_' + which_tab,
                            children=html.Div(['Drag and Drop the file here or ',
                                                html.A('Select Files')]),
                            style={'width': '100%',
                                   'height': '60px',
                                   'lineHeight': '60px',
                                   'borderWidth': '1px',
                                   'borderStyle': 'dashed',
                                   'borderRadius': '5px',
                                   'textAlign': 'center',
                                   'margin': '10px'},
                            # Allow multiple files to be uploaded
                            multiple=False)]),
                 ]),
             dbc.Col([
                  html.Div(id='output_'+ which_tab +'_inform',
                           children = original_df(which_tab = which_tab,
                                                  df = df))
             ])]
                   
    return children
                                 
layout = dbc.Container([    
    dbc.Row([
        dbc.Col(html.H1("Welcome to Chimei Quality Control Project",
                            className="text-center")
                    , className="mb-5 mt-5")
        ]),
    
    dbc.Row([
            html.Div(id='alignment-body', className='app-body', children=[
              html.Div(id='control-tab', className='control-tabs', children=[
                 dcc.Tabs(id="import_data_tab", value='deal',
                          children=[
                              dcc.Tab(label = 'Deal Data', value='deal',
                                      children = tab_content(which_tab = 'deal',
                                                             df = eval('deal'))),
                               dcc.Tab(label = 'Spec Data', value='spec',
                                        children = tab_content(which_tab = 'spec',
                                                               df = eval('spec'))),
                               dcc.Tab(label = 'Customer Data', value='customer',
                                        children = tab_content(which_tab = 'customer',
                                                               df = eval('customer'))),
                               dcc.Tab(label = 'Agent Data', value='agent',
                                        children = tab_content(which_tab = 'agent',
                                                               df = eval('agent'))),
                         ]
                     )]
            )], style = {'display': 'inline-block', 'width': '75%'})
    ]),
    
    dbc.Row(
        dbc.Col(
            html.Button(
                id='next_page',
                children=html.A(
                    children='Next',
                    href='/EDA',
                    style={'textDecoration':'none'})  
            ),
        align='end'
    ))
])

    