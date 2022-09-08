
import streamlit as st
import streamlit as st
import yfinance as yf
from datetime import datetime
from hydralit import HydraHeadApp

#create a wrapper class
class NewsApp(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
        st.title('Financial News')