import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
import altair as alt
import pandas as pd
# from data.create_data import create_table
from datetime import date
today = date.today()
# Raw Package
import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go

today = date.today()

#add an import to Hydralit
from hydralit import HydraHeadApp

#create a wrapper class
class ForexApp(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
        from tradingview_ta import TA_Handler, Interval
        import streamlit as st
        import tradingview_ta, requests, os
        from datetime import timezone
        from st_aggrid import AgGrid
        import pandas as pd
        import yfinance as yf
        from streamlit_echarts import st_echarts
        import matplotlib.pyplot as plt

        def color_negative_red(val):
            color = 'red' if val == 'SELL' else 'green' if val == 'BUY' else 'green' if val == 'STRONG_BUY' else 'red' if val == 'STRONG_SELL' else 'white'
            return 'color: %s' % color

        def get_analysis(symbol: str, exchange: str, screener: str, interval: str):
            # TradingView Technical Analysis
            handler = tradingview_ta.TA_Handler()
            handler.set_symbol_as(symbol)
            handler.set_exchange_as_crypto_or_stock(exchange)
            handler.set_screener_as_stock(screener)
            handler.set_interval_as(interval)
            analysis = handler.get_analysis()

            return analysis

        st.title("Forex Trading Signals")

        currency_pair = pd.read_csv(
            '/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/pages/currency_pair.csv')

        symbol = st.selectbox('Select the Currency Pairs', currency_pair).upper()

        exchange = "FX_IDC"
        screener = "forex"

        interval = st.selectbox("Interval", ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

        tesla = TA_Handler()
        tesla.set_symbol_as(symbol)
        tesla.set_exchange_as_crypto_or_stock(exchange)
        tesla.set_screener_as_stock(screener)
        tesla.set_interval_as(interval)

        analysis = get_analysis(symbol, exchange, screener, interval)
        st.markdown("Success!")

        '''
        st.title("Symbol: `" + analysis.symbol + "`")
        
        st.markdown("Exchange: `" + analysis.exchange + "`")
        st.markdown("Screener: `" + analysis.screener + "`")
        

        st.header("Interval: `" + analysis.interval + "`")
        

        if analysis.time and analysis.time.astimezone():
            st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")
            '''

        col14, col15 = st.columns(2)
        with col14:
            st.header("Symbol: `" + analysis.symbol + "`")
        with col15:
            st.header("Interval: `" + analysis.interval + "`")



        st.header("Summary Of Indicators")
        if analysis.time and analysis.time.astimezone():
            st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")

        a = tesla.get_analysis().summary

        col10, col12, col13 = st.columns(3)
        col10.metric('RECOMMENDATION', a['RECOMMENDATION'])
        col12.metric("BUY", a['BUY'])
        col13.metric("SELL", a['SELL'])

        tesla1 = TA_Handler()
        tesla1.set_symbol_as(symbol)
        tesla1.set_exchange_as_crypto_or_stock(exchange)
        tesla1.set_screener_as_stock(screener)
        tesla1.set_interval_as("1m")


        tesla2 = TA_Handler()
        tesla2.set_symbol_as(symbol)
        tesla2.set_exchange_as_crypto_or_stock(exchange)
        tesla2.set_screener_as_stock(screener)
        tesla2.set_interval_as("5m")


        tesla3 = TA_Handler()
        tesla3.set_symbol_as(symbol)
        tesla3.set_exchange_as_crypto_or_stock(exchange)
        tesla3.set_screener_as_stock(screener)
        tesla3.set_interval_as("15m")


        tesla4 = TA_Handler()
        tesla4.set_symbol_as(symbol)
        tesla4.set_exchange_as_crypto_or_stock(exchange)
        tesla4.set_screener_as_stock(screener)
        tesla4.set_interval_as("30m")


        tesla5 = TA_Handler()
        tesla5.set_symbol_as(symbol)
        tesla5.set_exchange_as_crypto_or_stock(exchange)
        tesla5.set_screener_as_stock(screener)
        tesla5.set_interval_as("1h")


        tesla6 = TA_Handler()
        tesla6.set_symbol_as(symbol)
        tesla6.set_exchange_as_crypto_or_stock(exchange)
        tesla6.set_screener_as_stock(screener)
        tesla6.set_interval_as("4h")


        tesla7 = TA_Handler()
        tesla7.set_symbol_as(symbol)
        tesla7.set_exchange_as_crypto_or_stock(exchange)
        tesla7.set_screener_as_stock(screener)
        tesla7.set_interval_as("1d")

        tesla8 = TA_Handler()
        tesla8.set_symbol_as(symbol)
        tesla8.set_exchange_as_crypto_or_stock(exchange)
        tesla8.set_screener_as_stock(screener)
        tesla8.set_interval_as("1W")

        tesla9 = TA_Handler()
        tesla9.set_symbol_as(symbol)
        tesla9.set_exchange_as_crypto_or_stock(exchange)
        tesla9.set_screener_as_stock(screener)
        tesla9.set_interval_as("1M")


        my_expander1 = st.expander(label='Real Time Recommendation - Expand me')
        with my_expander1:
            cola, col0, col1, col2 = st.columns(4)
            cola.header("Date")
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            cola.write(today)
            col0.header("Currency Pair")
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col0.write(symbol)
            col1.header("TimeFrame")
            col1.write("1 Minute")
            col1.write("5 Minutes")
            col1.write("15 Minutes")
            col1.write("30 Minute")
            col1.write("1 Hour")
            col1.write("4 Hours")
            col1.write("1 Day")
            col1.write("1 Week")
            col1.write("1 Month")
            col2.header("Signals")
            col2.write(tesla1.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla2.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla3.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla4.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla5.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla6.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla7.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla8.get_analysis().summary["RECOMMENDATION"])
            col2.write(tesla9.get_analysis().summary["RECOMMENDATION"])







        b = tesla.get_analysis().oscillators
        df1 = pd.DataFrame(b)
        df1 = df1.style.applymap(color_negative_red)



        d = tesla.get_analysis().moving_averages
        df4 = pd.DataFrame(d)
        df4 = df4.style.applymap(color_negative_red)


        my_expander = st.expander(label='Indicators - Expand me')
        with my_expander:
            col1, col2 = st.columns(2)
            col1.header("Moving Averages")
            col1.table(df4)
            col2.header("Oscillators")
            col2.table(df1)

        my_expander2 = st.expander(label='Live Historical Data - Expand me')
        tick = (symbol + "=X")

        data = yf.download(tickers=tick, period="1d", interval=interval)
        df = pd.DataFrame(data)
        df = df.reset_index()
        df = df.set_index('Datetime')

        with my_expander2:
            st.dataframe(df)


        my_expander3 = st.expander(label='Chart - Expand me')
        with my_expander3:

            st.line_chart(df["Close"])

            st.write("")

            st.bar_chart(df["Close"])
















