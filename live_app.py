import os, sys, json, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from businessdate import BusinessDate
# import stock_info module from yahoo_fin
from yahoo_fin import stock_info as si
from streamlit_autorefresh import st_autorefresh

# Data Source
import yfinance as yf

# Data viz
import plotly.graph_objs as go
import json
from datetime import date
import datetime
today = date.today()
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import plotly.graph_objects as go
from st_utils import show_plotly
from yf_utils import tickers_parser, get_dfs_by_tickers
from plotly_utils \
import plotly_ohlc_chart
from ta_utils import add_ATR, add_KER

from stock_returns import get_yf_data
import streamlit as st
import streamlit as st
import yfinance as yf
from datetime import datetime





# or any other ticker


# get quote table back as a data frame
# st.write(si.get_quote_table("aapl", dict_result=False))

# or get it back as a dictionary (default)
# st.write(si.get_quote_table("aapl"))

# get quote table back as a data frame
#st.write(si.get_quote_table("aapl", dict_result=False))

# or get it back as a dictionary (default)
#st.write(si.get_quote_table("aapl"))

from hydralit import HydraHeadApp

#create a wrapper class
class LiveApp(HydraHeadApp):
   def run(self):
        from tradingview_ta import TA_Handler, Interval
        import streamlit as st
        import tradingview_ta, requests, os
        from datetime import timezone
        from st_aggrid import AgGrid
        import pandas as pd
        import yfinance as yf
        from streamlit_echarts import st_echarts
        import matplotlib.pyplot as plt
        exchange = "FX_IDC"
        screener = "forex"

        def get_analysis(symbol):
            # TradingView Technical Analysis
            handler = tradingview_ta.TA_Handler()
            handler.set_symbol_as(symbol)
            handler.set_exchange_as_crypto_or_stock(exchange)
            handler.set_screener_as_stock(screener)
            handler.set_interval_as(interval)
            analysis = handler.get_analysis()

            return analysis


        # Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
        # after it's been refreshed 100 times.
        count = st_autorefresh(interval=120000, limit=100, key="fizzbuzzcounter")

        # The function returns a counter for number of refreshes. This allows the
        # ability to make special requests at different intervals based on the count
        if count == 0:
            st.write("Count is zero")
        elif count % 3 == 0 and count % 5 == 0:
            st.write("FizzBuzz")
        elif count % 3 == 0:
            st.write("Fizz")
        elif count % 5 == 0:
            st.write("Buzz")
        else:
            st.write(f"Count: {count}")

        my_expander1 = st.expander(label='Forex Top Currency Pairs - Expand me')
        with my_expander1:
            st.header("Summary Of Indicators")
            interval = st.selectbox("Interval", ("5m", "1m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

            st.subheader("Interval is : " + interval)


            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "EUR/USD")
            col31.metric("BUY", get_analysis('EURUSD').summary['BUY'])
            col32.metric("SELL", get_analysis('EURUSD').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('EURUSD').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('EURUSD').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "EUR/GBP")
            col31.metric("BUY", get_analysis('EURGBP').summary['BUY'])
            col32.metric("SELL", get_analysis('EURGBP').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('EURGBP').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('EURGBP').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/JPY")
            col31.metric("BUY", get_analysis('USDJPY').summary['BUY'])
            col32.metric("SELL", get_analysis('USDJPY').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDJPY').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDJPY').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "GBP/USD")
            col31.metric("BUY", get_analysis('GBPUSD').summary['BUY'])
            col32.metric("SELL", get_analysis('GBPUSD').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('GBPUSD').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('GBPUSD').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "AUD/USD")
            col31.metric("BUY", get_analysis('AUDUSD').summary['BUY'])
            col32.metric("SELL", get_analysis('AUDUSD').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('AUDUSD').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('AUDUSD').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/CAD")
            col31.metric("BUY", get_analysis('USDCAD').summary['BUY'])
            col32.metric("SELL", get_analysis('USDCAD').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDCAD').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDCAD').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/CNY")
            col31.metric("BUY", get_analysis('USDCNY').summary['BUY'])
            col32.metric("SELL", get_analysis('USDCNY').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDCNY').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDCNY').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/CHF")
            col31.metric("BUY", get_analysis('USDCHF').summary['BUY'])
            col32.metric("SELL", get_analysis('USDCHF').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDCHF').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDCHF').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/HKD")
            col31.metric("BUY", get_analysis('USDHKD').summary['BUY'])
            col32.metric("SELL", get_analysis('USDHKD').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDHKD').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDHKD').summary['RECOMMENDATION'])

            col30, col31, col32, col33, col34 = st.columns(5)
            col30.metric("CURRENCY PAIR", "USD/KRW")
            col31.metric("BUY", get_analysis('USDKRW').summary['BUY'])
            col32.metric("SELL", get_analysis('USDKRW').summary['SELL'])
            col33.metric("NEUTRAL", get_analysis('USDKRW').summary['NEUTRAL'])
            col34.metric('RECOMMENDATION', get_analysis('USDKRW').summary['RECOMMENDATION'])

            st.write("")

            symbls = ['USDCAD', 'EURJPY', 'EURUSD', 'EURCHF', 'USDCHF', 'EURGBP', 'GBPUSD', 'AUDCAD', 'NZDUSD',
                      'GBPCHF', 'AUDUSD', 'GBPJPY', 'USDJPY', 'CHFJPY', 'EURCAD', 'AUDJPY', 'EURAUD', 'AUDNZD']
            d = {}

            st.title("FOREX TRADING GUIDE")
            col51, col52, col53, col54 = st.columns(4)

            with col51:
                st.header("STRONG BUY")
                for symbl in symbls:
                    output = TA_Handler(
                        symbol=symbl,
                        screener="forex",
                        exchange="FX_IDC",
                        interval=interval)
                    d[symbl] = output.get_analysis().summary
                    for key in d[symbl]:
                        if d[symbl][key] == "STRONG_BUY":
                            st.write(symbl)

            with col52:
                st.header("STRONG SELL")
                for symbl in symbls:
                    output = TA_Handler(
                        symbol=symbl,
                        screener="forex",
                        exchange="FX_IDC",
                        interval=interval)
                    d[symbl] = output.get_analysis().summary
                    for key in d[symbl]:
                        if d[symbl][key] == "STRONG_SELL":
                            st.write(symbl)

            with col53:
                st.header("BUY")
                for symbl in symbls:
                    output = TA_Handler(
                        symbol=symbl,
                        screener="forex",
                        exchange="FX_IDC",
                        interval=interval)
                    d[symbl] = output.get_analysis().summary
                    for key in d[symbl]:
                        if d[symbl][key] == "BUY":
                            st.write(symbl)

            with col54:
                st.header("SELL")
                for symbl in symbls:
                    output = TA_Handler(
                        symbol=symbl,
                        screener="forex",
                        exchange="FX_IDC",
                        interval=interval)
                    d[symbl] = output.get_analysis().summary
                    for key in d[symbl]:
                        if d[symbl][key] == "SELL":
                            st.write(symbl)

        my_expander = st.expander(label='Crypto Forecaster - Expand me')
        with my_expander:
            st.title('Crypto Forecaster')
            st.markdown("This application enables you to predict on the future value of any cryptocurrency (available on Coinmarketcap.com), for \
                	any number of days into the future! The application is built with Streamlit (the front-end) and the Facebook Prophet model, \
                	which is an advanced open-source forecasting model built by Facebook, running under the hood. You can select to train the model \
                	on either all available data or a pre-set date range. Finally, you can plot the prediction results on both a normal and log scale.")

            ### Change sidebar color
            st.markdown(
                """
            <style>
            .sidebar .sidebar-content {
                background-image: linear-gradient(#D6EAF8,#D6EAF8);
                color: black;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            ### Set bigger font style
            st.markdown(
                """
            <style>
            .big-font {
                fontWeight: bold;
                font-size:22px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("Forecaster Settings")

            ### Select ticker & number of days to predict on
            selected_ticker = st.text_input("Select a ticker for prediction (i.e. BTC, ETH, LINK, etc.)", "BTC")
            period = int(
                st.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365,
                                        step=1))
            training_size = int(
                st.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

            ### Initialise scraper without time interval
            @st.cache
            def load_data(selected_ticker):
                init_scraper = CmcScraper(selected_ticker)
                df = init_scraper.get_dataframe()
                min_date = pd.to_datetime(min(df['Date']))
                max_date = pd.to_datetime(max(df['Date']))
                return min_date, max_date

            data_load_state = st.text('Loading data...')
            min_date, max_date = load_data(selected_ticker)
            data_load_state.text('Loading data... done!')

            ### Select date range
            date_range = st.selectbox("Select the timeframe to train the model on:",
                                              options=["All available data", "Specific date range"])

            if date_range == "All available data":

                ### Initialise scraper without time interval
                scraper = CmcScraper(selected_ticker)

            elif date_range == "Specific date range":

                ### Initialise scraper with time interval
                start_date = st.date_input('Select start date:', min_value=min_date, max_value=max_date,
                                                   value=min_date)
                end_date = st.date_input('Select end date:', min_value=min_date, max_value=max_date,
                                                 value=max_date)
                scraper = CmcScraper(selected_ticker, str(start_date.strftime("%d-%m-%Y")),
                                     str(end_date.strftime("%d-%m-%Y")))

            ### Pandas dataFrame for the same data
            data = scraper.get_dataframe()

            st.subheader('Raw data')
            st.write(data.head())

            ### Plot functions
            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
                fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

            def plot_raw_data_log():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
                fig.update_yaxes(type="log")
                fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

            ### Plot (log) data
            plot_log = st.checkbox("Plot log scale")
            if plot_log:
                plot_raw_data_log()
            else:
                plot_raw_data()

            ### Predict forecast with Prophet
            if st.button("Predict"):

                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

                ### Create Prophet model
                m = Prophet(
                    changepoint_range=training_size,  # 0.8
                    yearly_seasonality='auto',
                    weekly_seasonality='auto',
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',  # multiplicative/additive
                    changepoint_prior_scale=0.05
                )

                ### Add (additive) regressor
                for col in df_train.columns:
                    if col not in ["ds", "y"]:
                        m.add_regressor(col, mode="additive")

                m.fit(df_train)

                ### Predict using the model
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                ### Show and plot forecast
                st.subheader('Forecast data')
                st.write(forecast.head())

                st.subheader(f'Forecast plot for {period} days')
                fig1 = plot_plotly(m, forecast)
                if plot_log:
                    fig1.update_yaxes(type="log")
                st.plotly_chart(fig1)

                st.subheader("Forecast components")
                fig2 = m.plot_components(forecast)
                st.write(fig2)

        my_expander1 = st.expander(label='Crypto Forecaster - Expand me')
        with my_expander1:
            st.write("hello")

        my_expander2 = st.expander(label='Crypto Forecaster - Expand me')
        with my_expander2:
                st.write("hello")




        my_expander5 = st.expander(label='What to trade? - Expand me')
        with my_expander5:


                # The function returns a counter for number of refreshes. This allows the
                # ability to make special requests at different intervals based on the count


                st.title("FOREX TRADING GUIDE")
                '''
                st.selectbox("Select Interval", (
                "one_hour", "one_minute", "five_minutes", "fiften_minutes", "thirty_minutes", "two_hours", "four_hours",
                "one_day", "one_week", "one_month"))
                '''

                st.title("FOREX TRADING GUIDE")
                col51, col52, col53, col54 = st.columns(4)

                with col51:
                    st.header("STRONG BUY")
                    for symbl in symbls:
                        output = TA_Handler(
                            symbol=symbl,
                            screener="forex",
                            exchange="FX_IDC",
                            interval=interval)
                        d[symbl] = output.get_analysis().summary
                        for key in d[symbl]:
                            if d[symbl][key] == "STRONG_BUY":
                                st.write(symbl)

                with col52:
                    st.header("STRONG SELL")
                    for symbl in symbls:
                        output = TA_Handler(
                            symbol=symbl,
                            screener="forex",
                            exchange="FX_IDC",
                            interval=interval)
                        d[symbl] = output.get_analysis().summary
                        for key in d[symbl]:
                            if d[symbl][key] == "STRONG_SELL":
                                st.write(symbl)

                with col53:
                    st.header("BUY")
                    for symbl in symbls:
                        output = TA_Handler(
                            symbol=symbl,
                            screener="forex",
                            exchange="FX_IDC",
                            interval=interval)
                        d[symbl] = output.get_analysis().summary
                        for key in d[symbl]:
                            if d[symbl][key] == "BUY":
                                st.write(symbl)

                with col54:
                    st.header("SELL")
                    for symbl in symbls:
                        output = TA_Handler(
                            symbol=symbl,
                            screener="forex",
                            exchange="FX_IDC",
                            interval=interval)
                        d[symbl] = output.get_analysis().summary
                        for key in d[symbl]:
                            if d[symbl][key] == "SELL":
                                st.write(symbl)





        my_expander4 = st.expander(label='Crypto Forecaster - Expand me')
        with my_expander4:
            import datetime
            import pandas_datareader as pdr
            import cufflinks as cf

            APP_NAME = "Stock App!"

            # Page Configuration


            # Add some markdown

            st.markdown("# :chart_with_upwards_trend:")

            # Add app title
            st.title(APP_NAME)

            # List of tickers
            TICKERS = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT']

            # Select ticker
            ticker = st.selectbox('Select ticker', sorted(TICKERS), index=0)

            # Set start and end point to fetch data
            start_date = st.date_input('Start date', datetime.datetime(2021, 1, 1))
            end_date = st.date_input('End date', datetime.datetime.now().date())

            # Fetch the data for specified ticker e.g. AAPL from yahoo finance
            df_ticker = pdr.DataReader(ticker, 'yahoo', start_date, end_date)

            st.header(f'{ticker} Stock Price')

            if st.checkbox('Show raw data'):
                st.subheader('Raw data')
                st.write(df_ticker)

            # Interactive data visualizations using cufflinks
            # Create candlestick chart
            qf = cf.QuantFig(df_ticker, legend='top', name=ticker)

            # Technical Analysis Studies can be added on demand
            # Add Relative Strength Indicator (RSI) study to QuantFigure.studies
            qf.add_rsi(periods=20, color='java')

            # Add Bollinger Bands (BOLL) study to QuantFigure.studies
            qf.add_bollinger_bands(periods=20, boll_std=2, colors=['magenta', 'grey'], fill=True)

            # Add 'volume' study to QuantFigure.studies
            qf.add_volume()

            fig = qf.iplot(asFigure=True, dimensions=(800, 600))

            # Render plot using plotly_chart
            st.plotly_chart(fig)

        my_expander3 = st.expander(label='Crypto Forecaster - Expand me')
        with my_expander3:

            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas_datareader
            import pandas_datareader.data as web
            import datetime

            # import data
            # moving average

            st.title('Stock Prices for Different Timestamp')

            @st.cache
            def load_data(stock):
                data = web.get_data_yahoo(stock)
                return data

            def input_data():
                # choose stock
                # choose time frame
                # choose the date period
                stock = st.sidebar.selectbox('Stock', ['AAPL', 'BTC-USD', 'MSFT', 'TSLA', 'AMZN'])
                timeframe = st.sidebar.selectbox('Time Frame', ['Daily', 'Weekly', 'Monthly', 'Yearly'])

                timeframe_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Yearly': 'A'}
                # data = pd.read_csv('~/Downloads/' + stock + '.csv', parse_dates=True, index_col = 'Date')

                data = load_data(stock)
                NASDAQ = pandas_datareader.nasdaq_trader.get_nasdaq_symbols()

                st.header(NASDAQ[NASDAQ.index == stock][['Security Name']])

                start = st.sidebar.date_input('Starting date', data.index[0].date(), data.index[0].date(),
                                              data.index[-1].date())
                end = st.sidebar.date_input('Ending date', datetime.date.today(), data.index[0].date(),
                                            data.index[-1].date())

                data = data[(data.index.date >= start) & (data.index.date <= end)].dropna()

                return stock, data, timeframe, timeframe_dict

            def plot_graph(data, stock):
                # plot the closing price, returns, cum returns
                fig, ax = plt.subplots(3, 1, sharex=True)
                data.Close.plot(title=stock, ax=ax[0], ylabel="USD")
                ax[0].grid()

                data.Returns.mul(100).plot(title='Returns', ax=ax[1], ylabel='%')
                ax[1].grid()

                data['Cumulative Daily Returns'].plot(title='Cumulative Daily Returns', ax=ax[2], ylabel='%')

                ax[2].grid()

                plt.xticks(rotation=45)

                st.pyplot(fig)

            def MA(data, window):
                data_Close_rolled = data.Close.rolling(window, min_periods=1).mean()
                fig, ax = plt.subplots()
                data.Close.plot()
                data_Close_rolled.plot(label='{} MA'.format(window))
                plt.legend(loc='upper left')
                st.pyplot(fig)

            def MA_full(data, MA_type, window):
                if MA_type == 'SMA':
                    data_Close_rolled = data.Close.rolling(window=window, min_periods=1).mean()
                elif MA_type == 'EMA':
                    data_Close_rolled = data.Close.ewm(span=window, adjust=False).mean()
                elif MA_type == 'WMA':
                    weights = np.arange(1, (window + 1))
                    data_Close_rolled = data.Close.rolling(window).apply(
                        lambda prices: (np.dot(prices, weights) / weights.sum()),
                        raw=True)
                return data_Close_rolled

            def display_df(data):
                show = st.number_input('Enter number of last rows to view the stock data : ', 1, 100, 5, 5)

                st.dataframe(data.tail(int(show)))

                def main():
                    stock, data, timeframe, timeframe_dict = input_data()

                data = data.resample(timeframe_dict[timeframe]).mean()

                data.dropna(inplace=True)

                data['Returns'] = data.Close.pct_change()
                data['Returns'].fillna(0, inplace=True)

                # cum daily returns
                data['Cumulative Daily Returns'] = (1 + data.Returns).cumprod()

                data['Diff'] = data.Close - data.Open

                st.subheader(stock + '\'s Closing Prices & Returns')

                plot_graph(data, stock)

                st.subheader(stock + '\'s Closing Prices with Moving Average')

                MA_agree = st.sidebar.checkbox('Display Moving Average')

                if MA_agree:
                    # sigle or multi
                    num_agree = st.sidebar.selectbox('How many Moving Average ? ', ['Single', 'Multi'])
                    # if single : above
                    # choose type of MA
                    if num_agree == 'Single':
                        type_MA = st.sidebar.selectbox('Type of MA', ['SMA', 'WMA', 'EMA'])
                        window = st.sidebar.slider('window', 0, 250, 50)
                        data_Close_rolled = MA_full(data, type_MA, window)
                        fig, ax = plt.subplots()
                        data.Close.plot()
                        data_Close_rolled.plot(label='{} {}'.format(window, type_MA))
                        plt.legend(loc='upper left')
                        st.pyplot(fig)
                        # problem with WMA
                    # if multi
                    else:
                        choices = st.sidebar.multiselect('Which MA you would like to include ?', ['SMA', 'EMA', 'WMA'])
                        num_dict = {}
                        for choice in choices:
                            num = st.sidebar.selectbox('Number of {}'.format(choice), [0, 1, 2, 3, 4, 5])
                            counter = []
                            for i in range(1, (num + 1)):
                                counter.append(st.sidebar.slider('Window for {}\'s {}'.format(i, choice), 0, 250, 10))
                            num_dict[choice] = counter
                        if sum([len(num_dict[choice]) for choice in choices]) > 3:
                            st.warning('More than 3 MA')
                        else:
                            fig, ax = plt.subplots()
                            data.Close.plot(label='Closing Price')
                            for temp in num_dict.keys():
                                for j in num_dict[temp]:
                                    data_Close_rolled = MA_full(data, temp, j)
                                    data_Close_rolled.plot(label='{} {}'.format(j, temp))
                            plt.legend(loc='upper left')
                            st.pyplot(fig)
                        # choose type of MA for each MA
                        # sepecify number of MA wanted
                        # make sure total <= 3
                        # specify each MA's window length
                        # output a single interactive graph

                display_df(data)

            if __name__ == '__main__':
                main()
            ## where to go next ###

            # 1. forecasting using time series method such as Prophet, ARIMA, AR, MA, LSTM
            # 2. include simple strategy such as SMA, MACD, momentum ...
            # 1. backtest the strategy
            # 2. or input initial wealth at certain starting and ending date and display the possible change of wealth after certain strategy adopted
            # 3. display finance indicators or information for the stock prices
            # 4. change all the plots into interactive plots using plotly, bokeh, altair




