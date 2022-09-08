import streamlit as st
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
    stock = st.sidebar.selectbox('Stock', ['AAPL','BTC-USD','MSFT', 'TSLA', 'AMZN'])
    timeframe = st.sidebar.selectbox('Time Frame',['Daily', 'Weekly', 'Monthly', 'Yearly'])

    timeframe_dict = {'Daily':'D', 'Weekly' : 'W', 'Monthly':'M', 'Yearly': 'A'}
    #data = pd.read_csv('~/Downloads/' + stock + '.csv', parse_dates=True, index_col = 'Date')

    data = load_data(stock)
    NASDAQ = pandas_datareader.nasdaq_trader.get_nasdaq_symbols()

    st.header(NASDAQ[NASDAQ.index == stock][['Security Name']])

    start = st.sidebar.date_input('Starting date',  data.index[0].date(), data.index[0].date(), data.index[-1].date())
    end = st.sidebar.date_input('Ending date',  datetime.date.today(), data.index[0].date(), data.index[-1].date())

    data = data[(data.index.date >= start )& (data.index.date <= end)].dropna()

    return stock, data, timeframe, timeframe_dict

def plot_graph(data,stock):
    # plot the closing price, returns, cum returns
    fig, ax = plt.subplots(3,1,  sharex = True)
    data.Close.plot(title = stock, ax = ax[0], ylabel = "USD")
    ax[0].grid()

    data.Returns.mul(100).plot(title = 'Returns', ax = ax[1], ylabel = '%')
    ax[1].grid()

    data['Cumulative Daily Returns'].plot(title = 'Cumulative Daily Returns', ax = ax[2], ylabel = '%')

    ax[2].grid()

    plt.xticks(rotation = 45)

    st.pyplot(fig)

def MA(data, window):
    data_Close_rolled = data.Close.rolling(window, min_periods = 1).mean()
    fig, ax = plt.subplots()
    data.Close.plot()
    data_Close_rolled.plot(label = '{} MA'.format(window))
    plt.legend(loc = 'upper left')
    st.pyplot(fig)

def MA_full(data, MA_type, window):
    if MA_type == 'SMA':
        data_Close_rolled = data.Close.rolling(window = window, min_periods = 1).mean()
    elif MA_type == 'EMA':
        data_Close_rolled = data.Close.ewm(span = window, adjust = False).mean()
    elif MA_type == 'WMA':
        weights = np.arange(1, (window + 1))
        data_Close_rolled = data.Close.rolling(window).apply(lambda prices: (np.dot(prices, weights) / weights.sum()), raw = True)
    return data_Close_rolled

def display_df(data):
    show = st.number_input('Enter number of last rows to view the stock data : ', 1, 100 , 5, 5 )

    st.dataframe(data.tail(int(show)))

def main():
    stock, data, timeframe, timeframe_dict = input_data()

    data = data.resample(timeframe_dict[timeframe]).mean()

    data.dropna(inplace = True)

    data['Returns'] = data.Close.pct_change()
    data['Returns'].fillna(0, inplace = True)

    # cum daily returns
    data['Cumulative Daily Returns'] = (1+data.Returns).cumprod()

    data['Diff'] = data.Close - data.Open

    st.subheader(stock + '\'s Closing Prices & Returns')

    plot_graph(data,stock)

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
            data_Close_rolled.plot(label = '{} {}'.format(window, type_MA))
            plt.legend(loc = 'upper left')
            st.pyplot(fig)
            # problem with WMA
        # if multi
        else:
            choices = st.sidebar.multiselect('Which MA you would like to include ?', ['SMA', 'EMA', 'WMA'])
            num_dict = {}
            for choice in choices:
                num = st.sidebar.selectbox('Number of {}'.format(choice), [0,1,2,3,4,5])
                counter = []
                for i in range(1,(num+1)):
                    counter.append(st.sidebar.slider('Window for {}\'s {}'.format(i, choice), 0, 250, 10))
                num_dict[choice]=counter
            if sum([len(num_dict[choice]) for choice in choices]) > 3:
                st.warning('More than 3 MA')
            else:
                fig, ax = plt.subplots()
                data.Close.plot(label = 'Closing Price')
                for temp in num_dict.keys():
                    for j in num_dict[temp]:
                        data_Close_rolled = MA_full(data, temp, j)
                        data_Close_rolled.plot(label = '{} {}'.format(j, temp))
                plt.legend(loc = 'upper left')
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
       #1. backtest the strategy
       # 2. or input initial wealth at certain starting and ending date and display the possible change of wealth after certain strategy adopted
# 3. display finance indicators or information for the stock prices
# 4. change all the plots into interactive plots using plotly, bokeh, altair
