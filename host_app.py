from hydralit import HydraApp
import streamlit as st
from stocks_app import StocksApp
import hydralit_components as hc
from crypto_app import CryptoApp
from forex_app import ForexApp
from welcome_app import WelcomeApp
from pricing_app import PricingApp
from live_app import LiveApp
import hydralit_components as hc
from news_app import NewsApp
from contact_app import ContactApp
from backtesting_app import BacktestingApp
from stockanysis_app import StockanalysisApp
from prediction_app import Prediction_App


if __name__ == '__main__':

    #this is the host application, we add children to it and that's it!
    app = HydraApp(title='Sample Hydralit App',favicon="🐙")
  
    #add all your application classes here
    # The Home app, this is the default redirect if no target app is specified.

    app.add_app("Welcome", icon="🏠", app=WelcomeApp(), is_home=True)
    app.add_app("Forex", icon="🏠", app=ForexApp())
    app.add_app("Crypto", icon="🔊", app=CryptoApp())
    app.add_app("Stocks",icon="🔊", app=StocksApp())
    app.add_app("Pricing",icon="🔊", app=PricingApp())
    app.add_app("Analysis", icon="🔊", app=StockanalysisApp())
    app.add_app("Prediction", icon="🔊", app=Prediction_App())
    app.add_app("Live", icon="🔊", app=LiveApp())
    app.add_app("Backtesting", icon="🔊", app=BacktestingApp())
    app.add_app("Arbitrage", icon="🔊", app=ContactApp())



    complex_nav = {
        'Welcome': ['Welcome'],
        'Forex': ['Forex'],
        'Intro 🏆': ['Live', "Analysis"],
        'Arbitrage 🔥': ["Arbitrage"],
        'Arbitrage': ["Arbitrage"],
        'Crypto': ["Crypto"],
        'Stocks': ['Stocks']
    }

    #run the whole lot
    app.run()