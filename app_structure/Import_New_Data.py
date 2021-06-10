# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:13:03 2021

@author: acer
"""

import base64
from datetime import datetime
import io
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

import pandas as pd



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
                globals()['import_' + df_name] = df
                # globals()[df_name] = pd.concat([original_df, df], axis=0)
            except Exception as e:
                print(e)
                return html.Div([
                    dbc.Row(html.H4('Warning:')),
                    dbc.Row(html.Div([
                    '''The format of the new imported dataframe is wrong. Please check again!'''])),
                    dbc.Row(dash_table.DataTable(
                        data=original_df.tail(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in original_df.columns]),
                    )])

        
    except Exception as e:
        print(e)
        return html.Div([
                    dbc.Row(html.H4('Warning:')),
                    dbc.Row(html.Div([
                    '''There was an error processing this file! Please reload your data again.'''])),
                    dbc.Row(dash_table.DataTable(
                        data=original_df.tail(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in original_df.columns]),
                    )])
        
    return html.Div([
        dbc.Row(dbc.Card(
                    html.H3(children='Original Dataset',
                            className="text-center text-light bg-dark"), 
                body=True, color="dark")),   
        
        html.Br(),
        
        dbc.Row(dash_table.DataTable(
                    data=original_df.tail(10).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in original_df.columns]
                )),
        
        html.Br(),
        
        dbc.Row(dbc.Card(
                    html.H3(children='New Imported Dataset',
                            className="text-center text-light bg-dark"), 
                body=True, color="dark")),
        
        html.Br(),
        
        dbc.Row(html.H5(filename)),
        dbc.Row(html.H6('Upload Time:' + 
                datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S"))),
        
        dbc.Row(dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{'name':i, 'id':i} for i in df.columns]
                )),
        
        html.Hr(),
        
        html.Div(id=df_name + 'table_div',
                 children = [
                    dbc.Row(dbc.Card(
                            html.H3(children='New Concatenate Dataset',
                                    className="text-center text-light bg-dark"), 
                        body=True, color="dark")),
                    
                    html.Div(id=df_name + 'concat_table')
            ],
            style = {'display': 'none'}),
        
        dbc.Row(
            html.Button(
                children='Concatenate Two Dataframe',
                id=df_name + 'concat_df_button', 
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
            )),
    
        html.Hr(),  # horizontal line
    
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

# tab content for tabs
def tab_content(which_tab, df):

    children = []
    if isinstance(df, pd.DataFrame):
        children=html.Div([
                     dbc.Row(
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
                                multiple=False)),
                    
                    html.Div(id='output_'+ which_tab +'_inform'),
                ])
    else:
        warning_str = '''You haven't upload any data to ''' + which_tab + ' before!'
        children = [html.H4(warning_str)]
                 
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
                              dcc.Tab(id='deal_tab',
                                      label = 'Deal Data', value='deal'),
                               dcc.Tab(id='spec_tab',
                                       label = 'Spec Data', value='spec'),
                               dcc.Tab(id='customer_tab',
                                       label = 'Customer Data', value='customer'),
                               dcc.Tab(id='agent_tab',
                                       label = 'Agent Data', value='agent'),
                         ]
                     )]
            )], style = {'display': 'inline-block'})
            #, 'width': '75%'
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

    