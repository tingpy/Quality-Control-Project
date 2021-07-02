# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:50:13 2021

@author: acer
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import math


def generate_element(str):
    children = []
    cut = 2
    
    str_list = str.split('\n')
    for i in range(len(str_list)):
        key = str_list[i].split(' : ')[0]
        var = str_list[i].split(' : ')[1].split(', ')
        
        children.append(dbc.Row(html.P(key,style ={'color':'red'})))
        for j in range(math.ceil( (len(var)+1)/cut )):
            col = [dbc.Row(html.P(k,style ={'color':'red'})) for k in var[j*2 : (j+1)*2]]
            children.append(dbc.Row(col))
    
    return children
    
    
def create_info_card(dic, cnt):
    card_content = [
        dbc.CardHeader('File ' + str(cnt+1)),
        
        dbc.CardBody([
            dbc.Row(html.P("Date:", style={'font-weight': 'bold'})),
            dbc.Row(html.P("\t\t" + dic['Date'])),
            dbc.Row(html.P("Last Date for Spec Data:", style={'font-weight': 'bold'})),
            dbc.Row(html.P("\t\t" + dic['Duration for Spec Data'])),
            dbc.Row(html.P("Last Date for Deal Data:", style={'font-weight': 'bold'})),
            dbc.Row(html.P("\t\t" + dic['Duration for Deal Data'])),
            dbc.Row(html.P("Selected Material:", style={'font-weight': 'bold'})),
            dbc.Row(html.P("\t\t" + dic['Selected Material'])),
            dbc.Row(html.P("Input Variable:", style={'font-weight': 'bold'}))] 
            + generate_element(dic['Input Variable'])
            ),
        
        dbc.CardFooter([
            dbc.Col(
                dbc.Button('Choose This!',
                       id = {'type': 'dynamic-choose-button',
                             'index': cnt},
                       n_clicks = 0)),
            dbc.Col(
                dbc.Button('Remove This!',
                       id = {'type': 'dynamic-delete-button',
                             'index': cnt},
                       n_clicks = 0)),
            dbc.Col(
                dbc.Button('Recover This!',
                       id = {'type': 'dynamic-recover-button',
                             'index': cnt},
                       n_clicks = 0,
                       style = {'display': 'none'}))
            ]),
        ]
        
    
    children = dbc.Col(
        dbc.Card(
            id = {'type': 'dynamic-Card',
                  'index': cnt},
            children = card_content,
            color= '#240d33',#"black", 
            outline = True,
            inverse=True
           
        ), className="mb-4")
    
    return children

def warning_dialogue(cnt):
    if cnt == 0:
        return  '''You haven't chosen any import file!'''
    elif cnt == 1:
        return []
    else:
        return  '''More than one file have been selected, please select again'''

def nothing_available(arg):
    return html.H3('Nothing available, please go back previous page.')

layout = html.Div([
    dbc.Row(
        dbc.Col(
            html.Div(id='info_card'),
            )
        ),
    
    html.Div(
        dcc.ConfirmDialog(id='warning_dialogue')),
    
    dbc.Row([
        dbc.Col(
            dbc.Button("Go back previous page", 
                    href="/EDA",
                    color="primary",
                    className='mt-3')
            ),
        
        dbc.Col(
            dbc.Button("Confirm and Go to next page",
                    id='confirm_import',
                    color="primary",
                    className='mt-3',
                    n_clicks=0)
            ),
        ]),
    ])
