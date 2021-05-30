# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:20:05 2021

@author: acer
"""


import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_bootstrap_components as dbc


layout = html.Div([

    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to Chimei Quality Control Project",
                            className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='This app is created for Chimei Quality Control Project! '
                                     )
                    , className="mb-4")
        ]),

        dbc.Row([
            html.P(children='It consists of three steps to complete the whole process.')
        ]),
        
        dbc.Row([
            html.P(children='In the first step, you can import the new data in this step.')
            ]),
        
        dbc.Row([
            html.P(children='In the second step, you need to select the material and some important variables for the model.')
            ]),
        
        dbc.Row([
            html.P(children='In the last step, you can choose how to present the result for the model.')
            ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Step 1:',
                                                className="text-center"),
                                       html.H3(children="Import Data",
                                               className="text-center"),
                                       html.P(children='Import deal and QC data, and check the current data status.'),
                                       dbc.Button("Append New Data", 
                                                  href="/Import_New_Data",
                                                  color="primary",
                                                  className='mt-3'),],
                              body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Step 2:',
                                               className="text-center"),
                                       html.H3(children='Variable Selection',
                                               className='text-center'),
                                       html.P(children='Select the material you want, and the spec you want to focus on.'),
                                       dbc.Button("Select Data",
                                                  href="/EDA",
                                                  color="primary",
                                                  className="mt-3"),
                                        ],
                              body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Step 3:',
                                                className="text-center"),
                                       html.H3(children='Model Result',
                                                className="text-center"),
                                       html.P(children='The final result, choose the type of plot you want and export the plot.'),
                                       dbc.Button("Show The Result",
                                                  href="/Model",
                                                  color="primary",
                                                  className="mt-3"),

                                        ],
                              body=True, color="dark", outline=True)
                    , width=4, className="mb-4")
        ], className="mb-5"),
        

        html.A("Creator: Li-Heng Ting")

    ])

])

    
    
