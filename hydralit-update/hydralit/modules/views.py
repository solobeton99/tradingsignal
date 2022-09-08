#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json

def buttom(col):
    page_number = col.number_input(
        label=" ",
        min_value=1,
        max_value=200,
        step=1,
    )

    return page_number


def table(data):
    table = st.empty()
    table.write(data.to_html(), unsafe_allow_html=True)

    return None


def box_select_coin(col_position):
    path = 'helpers/CoinsID.json'
    df = pd.read_json(path)
    box_coins =  col_position.multiselect('', df['id'], help='Select coins...')

    if box_coins == []:
        box_coins = None
    else:
        box_coins = str(box_coins).replace('[', '').replace(']', '')

    return box_coins


def sidebar():
    st.sidebar.title("CryptosProyect")
    st.sidebar.markdown("## About The Project")
    st.sidebar.markdown("first project with Streamlit. \
                         Consuming data from the crypto market through CoinGecko API, the required data will be displayed. \
                         More visualizations and functionalities will be implemented in the future.")

    st.sidebar.markdown("[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)] \
                        (https://share.streamlit.io/luisarg03/streamlit-cryptocurrency/app/app.py)")

    st.sidebar.markdown("[![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)] \
                        (https://github.com/Luisarg03)")

    st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)] \
                        (https://www.linkedin.com/in/luisarg03/)")