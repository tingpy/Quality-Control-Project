# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:10:29 2021

@author: acer
"""


import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Homepage", href="/mainpage"),
        dbc.DropdownMenuItem("Import New Data", href="/Import_New_Data"),
        dbc.DropdownMenuItem("EDA", href="/EDA"),
        dbc.DropdownMenuItem("Model", href="/Model"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/chimei_logo.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Quality Control Project", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)




