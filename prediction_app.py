# load library
import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime

from app.data_sourcing import Data_Sourcing, data_update
from app.indicator_analysis import Indications
from app.graph import Visualization
from tensorflow.keras.models import load_model
import streamlit as st
import gc

from hydralit import HydraHeadApp

#create a wrapper class
class Prediction_App(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
          # print title of web app
        my_expander1 = st.expander(label='Automated Technical Analysis')
        with my_expander1:
            data_update()

            def main(app_data):
                st.set_page_config(layout="wide")
                indication = 'Predicted'

                st.sidebar.subheader('Asset:')
                asset = st.sidebar.selectbox('', ('Cryptocurrency', 'Index Fund', 'Forex', 'Futures', 'Stocks'),
                                             index=0)

                if asset in ['Index Fund', 'Forex', 'Futures', 'Stocks']:
                    exchange = 'Yahoo! Finance'
                    app_data.exchange_data(exchange)
                    if asset == 'Stocks':
                        assets = app_data.stocks
                    elif asset == 'Index Fund':
                        assets = app_data.indexes
                    elif asset == 'Futures':
                        assets = app_data.futures
                    elif asset == 'Forex':
                        assets = app_data.forex

                    st.sidebar.subheader(f'{asset}:')
                    equity = st.sidebar.selectbox('', assets)

                    if asset == 'Stocks':
                        currency = app_data.df_stocks[(app_data.df_stocks['Company'] == equity)]['Currency'].unique()[0]
                        market = \
                        app_data.df_stocks[(app_data.df_stocks['Company'] == equity)]['Currency_Name'].unique()[0]
                    elif asset == 'Index Fund':
                        currency = 'Pts'
                        market = None
                    elif asset == 'Futures':
                        currency = 'USD'
                        market = None
                    elif asset == 'Forex':
                        currency = app_data.df_forex[(app_data.df_forex['Currencies'] == equity)]['Currency'].unique()[
                            0]
                        market = app_data.df_forex[(app_data.df_forex['Currencies'] == equity)]['Market'].unique()[0]

                    st.sidebar.subheader('Interval:')
                    interval = st.sidebar.selectbox('',
                                                    ('5 Minute', '15 Minute', '30 Minute', '1 Hour', '1 Day', '1 Week'),
                                                    index=4)

                elif asset in ['Cryptocurrency']:
                    exchange = 'Binance'
                    app_data.exchange_data(exchange)
                    markets = app_data.markets

                    st.sidebar.subheader('Market:')
                    market = st.sidebar.selectbox('', markets)
                    app_data.market_data(market)
                    assets = app_data.assets
                    currency = app_data.currency

                    st.sidebar.subheader('Crypto:')
                    equity = st.sidebar.selectbox('', assets)

                    st.sidebar.subheader('Interval:')
                    interval = st.sidebar.selectbox('', (
                    '3 Minute', '5 Minute', '15 Minute', '30 Minute', '1 Hour', '6 Hour', '12 Hour', '1 Day', '1 Week'),
                                                    index=1)

                label = asset

                st.sidebar.subheader('Trading Volatility:')
                risk = st.sidebar.selectbox('', ('Low', 'Medium', 'High'))

                st.title(f'Automated Technical Analysis.')
                st.subheader(f'{label} Data Sourced from {exchange}.')
                st.info(f'Predicting...')

                future_price = 1
                analysis = Visualization(exchange, interval, equity, indication, action_model, price_model, market)
                analysis_day = Indications(exchange, '1 Day', equity, market)
                requested_date = analysis.df.index[-1]
                current_price = float(analysis.df['Adj Close'][-1])
                change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
                requested_prediction_price = float(analysis.requested_prediction_price)
                requested_prediction_action = analysis.requested_prediction_action

                risks = {'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]],
                         'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],
                         'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]], }
                buy_price = float(risks[risk][0])
                sell_price = float(risks[risk][1])

                change = f'{float(change):,.2f}'
                if exchange == 'Yahoo! Finance':
                    current_price = f'{float(current_price):,.2f}'
                    requested_prediction_price = f'{float(requested_prediction_price):,.2f}'
                    buy_price = f'{float(buy_price):,.2f}'
                    sell_price = f'{float(sell_price):,.2f}'
                else:
                    current_price = f'{float(current_price):,.8f}'
                    requested_prediction_price = f'{float(requested_prediction_price):,.8f}'
                    buy_price = f'{float(buy_price):,.8f}'
                    sell_price = f'{float(sell_price):,.8f}'

                if analysis.requested_prediction_action == 'Hold':
                    present_statement_prefix = 'off from taking any action with'
                    present_statement_suffix = ' at this time'
                else:
                    present_statement_prefix = ''
                    present_statement_suffix = ''

                accuracy_threshold = {analysis.score_action: 75., analysis.score_price: 75.}
                confidence = dict()
                for score, threshold in accuracy_threshold.items():
                    if float(score) >= threshold:
                        confidence[score] = f'*({score}% confident)*'
                    else:
                        confidence[score] = ''

                forcast_prefix = int(interval.split()[0]) * future_price
                if forcast_prefix > 1:
                    forcast_suffix = str(interval.split()[1]).lower() + 's'
                else:
                    forcast_suffix = str(interval.split()[1]).lower()

                asset_suffix = 'price'

                st.markdown(f'**Prediction Date & Time (UTC):** {str(requested_date)}.')
                st.header(f'**Current Price:** {currency} {current_price}.')
                st.subheader(f'**{interval} Price Change:** {change}%.')
                st.markdown(
                    f'**Recommended Trading Action:** You should **{requested_prediction_action.lower()}** {present_statement_prefix} this {label.lower()[:6]}{present_statement_suffix}. {str(confidence[analysis.score_action])}')
                st.markdown(
                    f'**Estimated Forecast Price:** The {label.lower()[:6]} {asset_suffix} for **{equity}** is estimated to be **{currency} {requested_prediction_price}** in the next **{forcast_prefix} {forcast_suffix}**. {str(confidence[analysis.score_price])}')
                if requested_prediction_action == 'Hold':
                    st.markdown(
                        f'**Recommended Trading Margins:** You should consider buying more **{equity}** {label.lower()[:6]} at **{currency} {buy_price}** and sell it at **{currency} {sell_price}**.')

                prediction_fig = analysis.prediction_graph(asset)
                if indication == 'Predicted':
                    testing_prefix = 'Predicted'
                else:
                    testing_prefix = 'Analysed'

                st.success(f'Historical {label[:6]} Price Action...({testing_prefix}.)')
                st.plotly_chart(prediction_fig, use_container_width=True)

                technical_analysis_fig = analysis.technical_analysis_graph()
                st.success(f'Technical Analysis results from the {label[:6]} Data...')
                st.plotly_chart(technical_analysis_fig, use_container_width=True)

            if __name__ == '__main__':
                import warnings
                import gc
                warnings.filterwarnings("ignore")
                gc.collect()
                action_model = load_model(
                    "/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/technical-analysis-master/models/action_prediction_model.h5")
                price_model = load_model(
                    "/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/technical-analysis-master/models/price_prediction_model.h5")
                app_data = Data_Sourcing()
                main(app_data=app_data)

        # print title of web app
        my_expander = st.expander(label='Stock Market Analysis and Prediction')
        with my_expander:
            st.title("Stock Market Analysis and Prediction")

            st.markdown(
                "> This app which predicts the future value of company stock or other ﬁnancial instrument traded on an exchange.")

            # Create a text element and let the reader know the data is loading.
            data_load_state = st.text('Loading data...')

            # Load data from yahoo finance.
            b = st.text_input('Enter Stock Symbol', 'AAPL')
            start = dt.date(2010, 1, 1)
            end = dt.date.today()
            data = pdr.get_data_yahoo(b, start, end)

            # fill nan vale with next value within columns
            data.fillna(method="ffill", inplace=True)

            # Notify the reader that the data was successfully loaded.
            data_load_state.text('Loading data...done!')

            # create checkbox
            if st.checkbox('Show raw data'):
                st.subheader('Raw data')
                st.write(data)

            # show the description of data
            st.subheader('Detail description about Datasets:-')
            descrb = data.describe()
            st.write(descrb)

            # create new columns like year, month, day
            data["Year"] = data.index.year
            data["Month"] = data.index.month
            data["Weekday"] = data.index.day_name()

            # dislay graph of open and close column
            st.subheader('Graph of Close & Open:-')
            st.line_chart(data[["Open", "Close"]])

            # display plot of Adj Close column in datasets
            st.subheader('Graph of Adjacent Close:-')
            st.line_chart(data['Adj Close'])

            # display plot of volume column in datasets
            st.subheader('Graph of Volume:-')
            st.line_chart(data['Volume'])

            # create new cloumn for data analysis.
            data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
            data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
            data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

            # display the new dataset after modificaton
            st.subheader('Newly format DataSet:-')
            st.dataframe(data.tail(500))

            forecast_col = 'Adj Close'
            forecast_out = int(math.ceil(0.01 * len(data)))
            data['label'] = data[forecast_col].shift(-forecast_out)

            X = np.array(data.drop(['label'], 1))
            X = preprocessing.scale(X)
            X_lately = X[-forecast_out:]
            X = X[:-forecast_out]
            data.dropna(inplace=True)
            y = np.array(data['label'])

            # split dataset into train and test dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf = LinearRegression(n_jobs=-1)
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)

            # display the accuracy of forecast value.
            st.subheader('Accuracy:')
            st.write(confidence)

            forecast_set = clf.predict(X_lately)
            data['Forecast'] = np.nan

            last_date = data.iloc[-1].name
            last_unix = last_date.timestamp()
            one_day = 86400
            next_unix = last_unix + one_day

            for i in forecast_set:
                next_date = datetime.datetime.fromtimestamp(next_unix)
                next_unix += 86400
                data.loc[next_date] = [np.nan for _ in range(len(data.columns) - 1)] + [i]
                last_date = data.iloc[-1].name
                dti = pd.date_range(last_date, periods=forecast_out + 1, freq='D')
                index = 1
            for i in forecast_set:
                data.loc[dti[index]] = [np.nan for _ in range(len(data.columns) - 1)] + [i]
                index += 1

            # display the forecast value.
            st.subheader('Forecast value :-')
            st.dataframe(data.tail(50))

            # display the graph of adj close and forecast columns
            st.subheader('Graph of Adj Close and Forecast :-')
            st.line_chart(data[["Adj Close", "Forecast"]])





