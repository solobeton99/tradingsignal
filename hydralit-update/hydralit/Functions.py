import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import performanceanalytics.statistics as pas
from scipy import stats

def import_scripts():
    tickers = pd.read_csv('/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/BacktestZone-main/scripts_list.csv')
    return tickers['SYMBOL']

def import_indicators():
    indicators = ['SuperTrend', '-DI, Negative Directional Index', 'Normalized Average True Range (NATR)', 'Average Directional Index (ADX)', 'Stochastic Oscillator Fast (SOF)', 'Stochastic Oscillator Slow (SOS)', 'Weighted Moving Average (WMA)', 'Momentum Indicator (MOM)', 'Vortex Indicator (VI)', 'Chande Momentum Oscillator (CMO)', 'Exponential Moving Average (EMA)', 'Triple Exponential Moving Average (TEMA)', 'Double Exponential Moving Average (DEMA)', 'Simple Moving Average (SMA)', 'Triangular Moving Average (TRIMA)', 'Chande Forecast Oscillator (CFO)', 'Choppiness Index', 'Aroon Down', 'Average True Range (ATR)', 'Williams %R', 'Parabolic SAR', 'Coppock Curve', '+DI, Positive Directional Index', 'Relative Strength Index (RSI)', 'MACD Signal', 'Aroon Oscillator', 'Stochastic RSI FastK', 'Stochastic RSI FastD', 'Ultimate Oscillator', 'Aroon Up', 'Bollinger Bands', 'TRIX', 'Commodity Channel Index (CCI)', 'MACD', 'MACD Histogram', 'Money Flow Index (MFI)']
    return indicators


def get_supertrend(high, low, close, lookback, multiplier):
    
        # ATR

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(lookback).mean()

        # H/L AVG AND BASIC UPPER & LOWER BAND

        hl_avg = (high + low) / 2
        upper_band = (hl_avg + multiplier * atr).dropna()
        lower_band = (hl_avg - multiplier * atr).dropna()

        # FINAL UPPER BAND

        final_bands = pd.DataFrame(columns = ['upper', 'lower'])
        final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
        final_bands.iloc[:,1] = final_bands.iloc[:,0]

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i,0] = 0
            else:
                if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                    final_bands.iloc[i,0] = upper_band[i]
                else:
                    final_bands.iloc[i,0] = final_bands.iloc[i-1,0]

        # FINAL LOWER BAND

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 1] = 0
            else:
                if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                    final_bands.iloc[i,1] = lower_band[i]
                else:
                    final_bands.iloc[i,1] = final_bands.iloc[i-1,1]

        # SUPERTREND

        supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
        supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]

        for i in range(len(supertrend)):
            if i == 0:
                supertrend.iloc[i, 0] = 0
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

        supertrend = supertrend.set_index(upper_band.index)
        supertrend = supertrend.dropna()[1:]

        # ST UPTREND/DOWNTREND

        upt = []
        dt = []
        close = close.iloc[len(close) - len(supertrend):]

        for i in range(len(supertrend)):
            if close[i] > supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close[i] < supertrend.iloc[i, 0]:
                upt.append(np.nan)
                dt.append(supertrend.iloc[i, 0])
            else:
                upt.append(np.nan)
                dt.append(np.nan)

        st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
        upt.index, dt.index = supertrend.index, supertrend.index

        return st
    
def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di.dropna(), minus_di.dropna(), adx_smooth.dropna()

def get_psar(high, low, close, af, max_af):
    psar = ta.psar(high = high, low = low, close = close, af = af, max_af = max_af).iloc[:,:2][2:]
    psar = psar.fillna(0)
    for i in range(len(psar)):
        if psar.iloc[i, 0] == 0:
            psar.iloc[i, 0] = psar.iloc[i, 1]
        else:
            pass
    return psar.iloc[:,0]

def implement_supertrend(num_stream, data, start_date, end_date):
    
    inputs1 = ['SuperTrend']
    inputs2 = ['Close', 'Open', 'High', 'Low', 'SuperTrend', 'Number']
    entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
    exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
    
    ######### SUPERTREND ENTRY CONDITION #########
   
    num_stream.sidebar.markdown('')
    entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)
    
    # 1. ST ENTRY DATA 1
    
    entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'st_entry_input1')
    
    if entry_input_1 == 'SuperTrend':
        period, multiplier = entry_condition_inputs.columns(2)
        period = int(period.text_input('SuperTrend Period', value = 7, key = 'st_entry_period1'))
        multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3, key = 'st_entry_multiplier1'))
        entry_data1 = get_supertrend(data['High'], data['Low'], data['Close'], period, multiplier)
        entry_data1.index = entry_data1.index.astype(str)
        entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
        entry_data1.index = pd.to_datetime(entry_data1.index)
    else:
        pass
    
    # 2. ST ENTRY COMPARATOR
    
    entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'st_entry_comparator')
    
    # 3. ST ENTRY DATA 2
    
    entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'st_entry_input2')
    
    if entry_input_2 == 'SuperTrend':
        period, multiplier = entry_condition_inputs.columns(2)
        period = int(period.text_input('SuperTrend Period', value = 7, key = 'st_entry_period2'))
        multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3, key = 'st_entry_multiplier2'))
        entry_data2 = get_supertrend(data['High'], data['Low'], data['Close'], period, multiplier)
        entry_data2.index = entry_data2.index.astype(str)
        entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
        entry_data2.index = pd.to_datetime(entry_data2.index)
    elif entry_input_2 == 'Number':
        entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
    else:
        entry_data2 = data[f'{entry_input_2}']
        entry_data2.index = entry_data2.index.astype(str)
        entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
        entry_data2.index = pd.to_datetime(entry_data2.index)
    
    ######### SUPERTREND EXIT CONDITION #########
    
    num_stream.sidebar.markdown('')
    exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)
    
    # 1. ST EXIT DATA 1
    
    exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'st_exit_input1')
    
    if exit_input_1 == 'SuperTrend':
        period, multiplier = exit_condition_inputs.columns(2)
        period = int(period.text_input('SuperTrend Period', value = 7, key = 'st_exit_period1'))
        multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3, key = 'st_exit_multiplier1'))
        exit_data1 = get_supertrend(data['High'], data['Low'], data['Close'], period, multiplier)
        exit_data1.index = exit_data1.index.astype(str)
        exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
        exit_data1.index = pd.to_datetime(exit_data1.index)
    else:
        pass
    
    # 2. ST EXIT COMPARATOR
    
    exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'st_exit_comparator')
    
    # 3. ST EXIT DATA 2
    
    exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'st_exit_input2')
    
    if exit_input_2 == 'SuperTrend':
        period, multiplier = exit_condition_inputs.columns(2)
        period = int(period.text_input('SuperTrend Period', value = 7, key = 'st_exit_period2'))
        multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3, key = 'st_exit_multiplier2'))
        exit_data2 = get_supertrend(data['High'], data['Low'], data['Close'], period, multiplier)
        exit_data2.index = exit_data2.index.astype(str)
        exit_data2 = entry_data2[exit_data2.index >= str(start_date)]
        exit_data2.index = pd.to_datetime(exit_data2.index)
    elif exit_input_2 == 'Number':
        exit_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
    else:
        exit_data2 = data[f'{exit_input_2}']
        exit_data2.index = exit_data2.index.astype(str)
        exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
        exit_data2.index = pd.to_datetime(exit_data2.index)
        
    return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_negative_directional_index(num_stream, data, start_date, end_date):
    
        inputs1 = ['-DI']
        inputs2 = ['+DI', 'ADX', '-DI', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### -DI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. -DI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = '-di_entry_input1')

        if entry_input_1 == '-DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_entry_offset1'))
            entry_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset).iloc[:,2]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. -DI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = '-di_entry_comparator')

        # 3. -DI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = '-di_entry_input2')

        if entry_input_2 == '+DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '+di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'ADX':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == '-DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass

        ######### -DI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. -DI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = '-di_exit_input1')

        if exit_input_1 == '-DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_exit_offset1'))
            exit_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. -DI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = '-di_exit_comparator')

        # 3. -DI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = '-di_exit_input2')

        if exit_input_2 == '+DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '+di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'ADX':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == '-DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_normalized_average_true_range(num_stream, data, start_date, end_date):

        inputs1 = ['NATR']
        inputs2 = ['NATR', 'TR', 'ATR', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### NATR ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. NATR ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'natr_entry_input1')

        if entry_input_1 == 'NATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 14, key = 'natr_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'natr_entry_offset1'))
            entry_data1 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. NATR ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'natr_entry_comparator')

        # 3. NATR ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'natr_entry_input2')

        if entry_input_2 == 'NATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'natr_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'natr_entry_offset2'))
            entry_data2 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TR':
            offset, period = entry_condition_inputs.columns(2)
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'tr_entry_offset2'))
            entry_data2 = ta.true_range(high = data.High, low = data.Low, close = data.Close, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'ATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'atr_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_entry_offset2'))
            entry_data2 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass

        ######## NATR EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. NATR EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'natr_exit_input1')

        if exit_input_1 == 'NATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 14, key = 'natr_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'natr_exit_offset1'))
            exit_data1 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. NATR EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'natr_exit_comparator')

        # 3. NATR EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'natr_exit_input2')

        if exit_input_2 == 'NATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'natr_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'natr_exit_offset2'))
            exit_data2 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TR':
            offset, period = exit_condition_inputs.columns(2)
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'tr_exit_offset2'))
            exit_data2 = ta.true_range(high = data.High, low = data.Low, close = data.Close, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'ATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'atr_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_exit_offset2'))
            exit_data2 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_average_directional_index(num_stream, data, start_date, end_date):
    
        inputs1 = ['ADX']
        inputs2 = ['+DI', '-DI', 'ADX', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### ADX ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. ADX ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'adx_entry_input1')

        if entry_input_1 == 'ADX':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_entry_offset1'))
            entry_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. ADX ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'adx_entry_comparator')

        # 3. ADX ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'adx_entry_input2')

        if entry_input_2 == '+DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 14, key = '+di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == '-DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'ADX':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass

        ######### ADX EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. ADX EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'adx_exit_input1')

        if exit_input_1 == 'ADX':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_exit_offset1'))
            exit_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. ADX EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'adx_exit_comparator')

        # 3. ADX EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'adx_exit_input2')

        if exit_input_2 == '+DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 14, key = '+di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == '-DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '-di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'ADX':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_stochastic_oscillator_fast(num_stream, data, start_date, end_date):
    
        inputs1 = ['SOF']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### SOF ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. SOF ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'sof_entry_input1')

        if entry_input_1 == 'SOF':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('SOF Period', value = 14, key = 'sof_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'sof_entry_offset1'))
            entry_data1 = ta.stoch(data.High, data.Low, data.Close, k = period, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. SOF ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'sof_entry_comparator')

        # 3. SOF ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'sof_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 30, key = 'number1')
        else:
            pass

       ######### SOF EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. SOF EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'sof_exit_input1')

        if exit_input_1 == 'SOF':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('SOF Period', value = 14, key = 'sof_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'sof_exit_offset1'))
            exit_data1 = ta.stoch(data.High, data.Low, data.Close, k = period, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. SOF EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'sof_exit_comparator')

        # 3. SOF EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'sof_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 80, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_stochastic_oscillator_slow(num_stream, data, start_date, end_date):
    
        inputs1 = ['SOS']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### sos ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. SOS ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'sos_entry_input1')

        if entry_input_1 == 'SOS':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('SOS Period', value = 3, key = 'sos_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'sos_entry_offset1'))
            entry_data1 = ta.stoch(data.High, data.Low, data.Close, d = period, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. SOS ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'sos_entry_comparator')

        # 3. SOS ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'sos_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 30, key = 'number1')
        else:
            pass

       ######### SOS EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. SOS EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'sos_exit_input1')

        if exit_input_1 == 'SOS':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('SOS Period', value = 3, key = 'sos_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'sos_exit_offset1'))
            exit_data1 = ta.stoch(data.High, data.Low, data.Close, d = period, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. SOS EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'sos_exit_comparator')

        # 3. SOS EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'sos_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 80, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_weighted_moving_average(num_stream, data, start_date, end_date):
    
        inputs1 = ['WMA']
        inputs2 = ['Close', 'EMA', 'SMA', 'TRIMA', 'TEMA', 'DEMA', 'WMA', 'Open', 'High', 'Low', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### WMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. WMA ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'wma_entry_input1')

        if entry_input_1 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 60, key = 'wma_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.wma(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. WMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'wma_entry_comparator')

        # 3. WMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'wma_entry_input2')

        if entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 20, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        if entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 20, key = 'ema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        if entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 20, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        if entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 20, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        if entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 20, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        if entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 20, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### WMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. WMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'wma_exit_input1')

        if exit_input_1 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 60, key = 'wma_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.wma(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. WMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'wma_exit_comparator')

        # 3. WMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'wma_exit_input2')

        if exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 20, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        if exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 20, key = 'ema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        if exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 20, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        if exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 20, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        if exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 20, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        if exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 20, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_momentum_indicator(num_stream, data, start_date, end_date):
    
        inputs1 = ['MOM']
        inputs2 = ['Number', 'MOM']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### MOM ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. MOM ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'mom_entry_input1')

        if entry_input_1 == 'MOM':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('MOM Period', value = 14, key = 'mom_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mom_entry_offset1'))
            entry_data1 = ta.mom(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. MOM ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'mom_entry_comparator')

        # 3. MOM ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'mom_entry_input2')

        if entry_input_2 == 'MOM':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('MOM Period', value = 14, key = 'mom_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mom_entry_offset2'))
            entry_data2 = ta.mom(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'mom_number1')
        else:
            pass

        ######### MOM EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. MOM EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'mom_exit_input1')

        if exit_input_1 == 'MOM':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('MOM Period', value = 14, key = 'mom_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mom_exit_offset1'))
            exit_data1 = ta.mom(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. MOM EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'mom_exit_comparator')

        # 3. MOM EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'mom_exit_input2')

        if exit_input_2 == 'MOM':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('MOM Period', value = 14, key = 'mom_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mom_exit_offset'))
            exit_data2 = ta.mom(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'mom_number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_vortex_indicator(num_stream, data, start_date, end_date):
    
        inputs1 = ['+VI', '-VI']
        inputs2 = ['-VI', '+VI', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### VI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. VI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'vi_entry_input1')

        if entry_input_1 == '+VI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('+VI Period', value = 14, key = '+vi_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+vi_entry_offset1'))
            entry_data1 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        elif entry_input_1 == '-VI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-VI Period', value = 14, key = '-vi_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-vi_entry_offset1'))
            entry_data1 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. VI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'vi_entry_comparator')

        # 3. VI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'vi_entry_input2')

        if entry_input_2 == '-VI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-VI Period', value = 14, key = '-vi_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-vi_entry_offset2'))
            entry_data2 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == '+VI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('+VI Period', value = 14, key = '+vi_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+vi_entry_offset2'))
            entry_data2 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass

        ######### VI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. VI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'vi_exit_input1')

        if exit_input_1 == '+VI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('+VI Period', value = 14, key = '+vi_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+vi_exit_offset1'))
            exit_data1 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        elif exit_input_1 == '-VI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-VI Period', value = 14, key = '-vi_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-vi_exit_offset1'))
            exit_data1 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. VI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'vi_exit_comparator')

        # 3. VI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'vi_exit_input2')

        if exit_input_2 == '-VI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-VI Period', value = 14, key = '-vi_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '-vi_exit_offset2'))
            exit_data2 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == '+VI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('+VI Period', value = 14, key = '+vi_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+vi_exit_offset2'))
            exit_data2 = ta.vortex(data.High, data.Low, data.Close, length = period, offset = offfset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_chande_momentum_oscillator(num_stream, data, start_date, end_date):

        inputs1 = ['CMO']
        inputs2 = ['Number', 'CMO']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### CMO ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. CMO ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'cmo_entry_input1')

        if entry_input_1 == 'CMO':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('CMO Period', value = 9, key = 'cmo_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cmo_entry_offset1'))
            entry_data1 = ta.cmo(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. CMO ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'cmo_entry_comparator')

        # 3. CMO ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'cmo_entry_input2')

        if entry_input_2 == 'CMO':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('CMO Period', value = 9, key = 'cmo_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cmo_entry_offset2'))
            entry_data2 = ta.cmo(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'cmo_number1')
        else:
            pass

        ######### CMO EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. CMO EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'cmo_exit_input1')

        if exit_input_1 == 'CMO':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CMO Period', value = 9, key = 'cmo_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cmo_exit_offset1'))
            exit_data1 = ta.cmo(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. CMO EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'cmo_exit_comparator')

        # 3. CMO EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'cmo_exit_input2')

        if exit_input_2 == 'CMO':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CMO Period', value = 9, key = 'cmo_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cmo_exit_offset'))
            exit_data2 = ta.cmo(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'cmo_number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_exponential_moving_average(num_stream, data, start_date, end_date):

        inputs1 = ['EMA']
        inputs2 = ['EMA', 'SMA', 'TRIMA', 'TEMA', 'DEMA', 'WMA', 'Open', 'High', 'Low', 'Close', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### EMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. EMA ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'ema_entry_input1')

        if entry_input_1 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 21, key = 'ema_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.ema(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. EMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'ema_entry_comparator')

        # 3. EMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'ema_entry_input2')

        if entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 50, key = 'ema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 50, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 50, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 50, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 50, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 50, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### EMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. EMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'ema_exit_input1')

        if exit_input_1 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 21, key = 'ema_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.ema(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. EMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'ema_exit_comparator')

        # 3. EMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'ema_exit_input2')

        if exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 50, key = 'ema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 50, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 50, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 50, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 50, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 50, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_triple_exponential_moving_average(num_stream, data, start_date, end_date):

        inputs1 = ['TEMA']
        inputs2 = ['TEMA', 'SMA', 'TRIMA', 'EMA', 'DEMA', 'WMA', 'Open', 'High', 'Low', 'Close', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### TEMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. TEMA ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'tema_entry_input1')

        if entry_input_1 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('tema Period', value = 20, key = 'tema_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.tema(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. TEMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'tema_entry_comparator')

        # 3. TEMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'tema_entry_input2')

        if entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 40, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 40, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 40, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 40, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 40, key = 'ema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 40, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### TEMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. TEMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'tema_exit_input1')

        if exit_input_1 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('tema Period', value = 20, key = 'tema_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.tema(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. TEMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'tema_exit_comparator')

        # 3. TEMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'tema_exit_input2')

        if exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 40, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 40, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 40, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 40, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 40, key = 'ema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 40, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
            
def implement_double_exponential_moving_average(num_stream, data, start_date, end_date):

        inputs1 = ['DEMA']
        inputs2 = ['DEMA', 'High', 'Low', 'Open', 'Close', 'Number', 'TEMA', 'SMA', 'TRIMA', 'EMA', 'WMA']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### DEMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. dema ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'dema_entry_input1')

        if entry_input_1 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 9, key = 'dema_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.dema(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. DEMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'dema_entry_comparator')

        # 3. DEMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'dema_entry_input2')

        if entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 21, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 21, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 21, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 21, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 21, key = 'ema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 21, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### DEMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. DEMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'dema_exit_input1')

        if exit_input_1 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 9, key = 'dema_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.dema(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. DEMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'dema_exit_comparator')

        # 3. DEMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'dema_exit_input2')

        if exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 21, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 21, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 21, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 21, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 21, key = 'ema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 21, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_simple_moving_average(num_stream, data, start_date, end_date):

        inputs1 = ['SMA']
        inputs2 = ['SMA', 'EMA', 'TRIMA', 'TEMA', 'DEMA', 'WMA', 'Open', 'High', 'Low', 'Close', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### SMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. SMA ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'sma_entry_input1')

        if entry_input_1 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 12, key = 'sma_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.sma(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. SMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'sma_entry_comparator')

        # 3. SMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'sma_entry_input2')

        if entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 26, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 26, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 26, key = 'sma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 26, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 26, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 26, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### SMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. SMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'sma_exit_input1')

        if exit_input_1 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 12, key = 'sma_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.sma(data.Close, length = period)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. SMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'sma_exit_comparator')

        # 3. SMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'sma_exit_input2')

        if exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 26, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 26, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 26, key = 'sma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 26, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 26, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 26, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_triangular_moving_average(num_stream, data, start_date, end_date):

        inputs1 = ['TRIMA']
        inputs2 = ['TRIMA', 'EMA', 'SMA', 'TEMA', 'DEMA', 'WMA', 'Open', 'High', 'Low', 'Close', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### TRIMA ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. TRIMA ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'trima_entry_input1')

        if entry_input_1 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 20, key = 'trima_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset0'))
            entry_data1 = ta.trima(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. TRIMA ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'trima_entry_comparator')

        # 3. TRIMA ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'trima_entry_input2')

        if entry_input_2 == 'TRIMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 50, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.trima(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'WMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 50, key = 'wma_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.wma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'EMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 50, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.ema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'SMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 50, key = 'trima_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.sma(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 50, key = 'tema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.tema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'DEMA':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 50, key = 'dema_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset1'))
            entry_data2 = ta.dema(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)

        ######### TRIMA EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. TRIMA EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'trima_exit_input1')

        if exit_input_1 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 20, key = 'trima_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'exit_offset'))
            exit_data1 = ta.trima(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. TRIMA EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'trima_exit_comparator')

        # 3. TRIMA EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'trima_exit_input2')

        if exit_input_2 == 'TRIMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TRIMA Period', value = 50, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.trima(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'WMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('WMA Period', value = 50, key = 'wma_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.wma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'EMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('EMA Period', value = 50, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.ema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'SMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('SMA Period', value = 50, key = 'trima_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.sma(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('TEMA Period', value = 50, key = 'tema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.tema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'DEMA':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('DEMA Period', value = 50, key = 'dema_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'offset2'))
            exit_data2 = ta.dema(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
            
def implement_chande_forecast_oscillator(num_stream, data, start_date, end_date):

        inputs1 = ['CFO']
        inputs2 = ['Number', 'CFO']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### CFO ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. CFO ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'cfo_entry_input1')

        if entry_input_1 == 'CFO':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('CFO Period', value = 14, key = 'cfo_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cfo_entry_offset1'))
            entry_data1 = ta.cfo(data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. CFO ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'cfo_entry_comparator')

        # 3. CFO ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'cfo_entry_input2')
        
        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        elif entry_input_2 == 'CFO':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('cfo Period', value = 25, key = 'cfo_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cfo_entry_offset2'))
            entry_data2 = ta.cfo(data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        else:
            pass

        ######## CFO EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. CFO EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'cfo_exit_input1')

        if exit_input_1 == 'CFO':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CFO Period', value = 14, key = 'cfo_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cfo_exit_offset1'))
            exit_data1 = ta.cfo(data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. CFO EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'cfo_exit_comparator')

        # 3. CFO EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'cfo_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        elif exit_input_2 == 'CFO':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CFO Period', value = 25, key = 'cfo_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cfo_exit_offset2'))
            exit_data2 = ta.cfo(data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_choppiness_index(num_stream, data, start_date, end_date):

        inputs1 = ['Choppiness Index']
        inputs2 = ['Number', 'Choppiness Index']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### CHOP ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. CHOP ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'ci_entry_input1')

        if entry_input_1 == 'Choppiness Index':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('CI Period', value = 14, key = 'ci_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'ci_entry_offset1'))
            entry_data1 = ta.chop(data.High, data.Low, data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. CHOP ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'ci_entry_comparator')

        # 3. CHOP ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'ci_entry_input2')
        
        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 40, key = 'number1')
        elif entry_input_2 == 'Choppiness Index':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('ci Period', value = 50, key = 'ci_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'ci_entry_offset2'))
            entry_data2 = ta.chop(data.High, data.Low, data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        else:
            pass

        ######## CHOP EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. CHOP EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'ci_exit_input1')

        if exit_input_1 == 'Choppiness Index':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CI Period', value = 14, key = 'ci_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'ci_exit_offset1'))
            exit_data1 = ta.chop(data.High, data.Low, data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. CHOP EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'ci_exit_comparator')

        # 3. CHOP EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'ci_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 60, key = 'number2')
        elif exit_input_2 == 'Choppiness Index':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CI Period', value = 50, key = 'ci_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'ci_exit_offset2'))
            exit_data2 = ta.chop(data.High, data.Low, data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_aroon_down(num_stream, data, start_date, end_date):

        inputs1 = ['Aroon Down']
        inputs2 = ['Aroon Up', 'Aroon Down', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### AROOND ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. AROOND ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'aroond_entry_input1')

        if entry_input_1 == 'Aroon Down':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroond_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_entry_offset1'))
            entry_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. AROOND ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'aroond_entry_comparator')

        # 3. AROOND ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'aroond_entry_input2')
        
        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        elif entry_input_2 == 'Aroon Up':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroond_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Aroon Down':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroond_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        else:
            pass

        ######## AROOND EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. AROOND EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'aroond_exit_input1')

        if exit_input_1 == 'Aroon Down':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroond_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_exit_offset1'))
            exit_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. AROOND EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'aroond_exit_comparator')

        # 3. AROOND EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'aroond_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        elif exit_input_2 == 'Aroon Up':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroond_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Aroon Down':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroond_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroond_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_average_true_range(num_stream, data, start_date, end_date):

        inputs1 = ['ATR']
        inputs2 = ['ATR', 'TR', 'NATR', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### atr ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. ATR ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'atr_entry_input1')

        if entry_input_1 == 'ATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('ATR Period', value = 14, key = 'atr_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_entry_offset1'))
            entry_data1 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. ATR ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'atr_entry_comparator')

        # 3. ATR ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'atr_entry_input2')

        if entry_input_2 == 'ATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('ATR Period', value = 25, key = 'atr_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_entry_offset2'))
            entry_data2 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'TR':
            offset, period = entry_condition_inputs.columns(2)
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'tr_entry_offset2'))
            entry_data2 = ta.true_range(high = data.High, low = data.Low, close = data.Close, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'NATR':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'atr_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_entry_offset2'))
            entry_data2 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass

        ######## ATR EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. ATR EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'atr_exit_input1')

        if exit_input_1 == 'ATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('ATR Period', value = 14, key = 'atr_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_exit_offset1'))
            exit_data1 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. ATR EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'atr_exit_comparator')

        # 3. ATR EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'atr_exit_input2')

        if exit_input_2 == 'ATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('ATR Period', value = 25, key = 'atr_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_exit_offset2'))
            exit_data2 = ta.atr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'TR':
            offset, period = exit_condition_inputs.columns(2)
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'tr_exit_offset2'))
            exit_data2 = ta.true_range(high = data.High, low = data.Low, close = data.Close, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'NATR':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('NATR Period', value = 25, key = 'atr_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'atr_exit_offset2'))
            exit_data2 = ta.natr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_williamsr(num_stream, data, start_date, end_date):

        inputs1 = ['Williams %R']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '>, Crossing Up', '==, Equal To']

        ######### WR ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. WR ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'wr_entry_input1')

        if entry_input_1 == 'Williams %R':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('W%R Period', value = 14, key = 'wr_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'wr_entry_offset1'))
            entry_data1 = ta.willr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. WR ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'wr_entry_comparator')

        # 3. WR ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'wr_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = -80, min_value = -100, 
                                                              max_value = 0, key = 'number1')
        else:
            pass

        ######## WR EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. WR EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'wr_exit_input1')

        if exit_input_1 == 'Williams %R':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('W%R Period', value = 14, key = 'wr_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'wr_exit_offset1'))
            exit_data1 = ta.willr(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. WR EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'wr_exit_comparator')

        # 3. WR EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'wr_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = -100, value = -20, 
                                                            max_value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_parabolic_sar(num_stream, data, start_date, end_date):

        inputs1 = ['Parabolic SAR']
        inputs2 = ['Close', 'High', 'Low', 'Open', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### PSAR ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. PSAR ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'psar_entry_input1')

        if entry_input_1 == 'Parabolic SAR':
            min_af, max_af = entry_condition_inputs.columns(2)
            min_af = float(min_af.text_input('Min AF', value = 0.02, key = 'psar_entry_af1'))
            max_af = float(max_af.text_input('Max AF', value = 0.2, key = 'psar_entry_maxaf1'))
            entry_data1 = get_psar(high = data.High, low = data.Low, close = data.Close, af = min_af, max_af = max_af)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. PSAR ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'psar_entry_comparator')

        # 3. PSAR ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'psar_entry_input2')

        if entry_input_2 == 'Parabolic SAR':
            min_af, max_af = entry_condition_inputs.columns(2)
            min_af = float(min_af.text_input('Min AF', value = '0.02', key = 'psar_entry_af2'))
            max_af = float(max_af.text_input('Max AF', value = '0.2', key = 'psar_entry_maxaf2'))
            entry_data2 = get_psar(high = data.High, low = data.Low, close = data.Close, af = min_af, max_af = max_af)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            entry_data2 = data[f'{entry_input_2}']
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
            
        ######### PSAR EXIT CONDITION #########
        
        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)
        
        # 1. PSAR EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'psar_exit_input1')

        if exit_input_1 == 'Parabolic SAR':
            min_af, max_af = exit_condition_inputs.columns(2)
            min_af = float(min_af.text_input('Min AF', value = 0.02, key = 'psar_exit_af1'))
            max_af = float(max_af.text_input('Max AF', value = 0.2, key = 'psar_exit_maxaf1'))
            exit_data1 = get_psar(high = data.High, low = data.Low, close = data.Close, af = min_af, max_af = max_af)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. PSAR EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'psar_exit_comparator')

        # 3. PSAR EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'psar_exit_input2')

        if exit_input_2 == 'Parabolic SAR':
            min_af, max_af = exit_condition_inputs.columns(2)
            min_af = float(min_af.text_input('Min AF', value = 0.02, key = 'psar_exit_af2'))
            max_af = float(max_af.text_input('Max AF', value = 0.2, key = 'psar_exit_maxaf2'))
            exit_data2 = get_psar(high = data.High, low = data.Low, close = data.Close, af = min_af, max_af = max_af)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            exit_data2 = data[f'{exit_input_2}']
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
            
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
            
def implement_coppock_curve(num_stream, data, start_date, end_date):

        inputs1 = ['Coppock Curve']
        inputs2 = ['Number', 'Coppock Curve']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### CC ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. CC ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'coppock_curve_entry_input1')

        if entry_input_1 == 'Coppock Curve':
            fast, slow = entry_condition_inputs.columns(2)
            period, offset = entry_condition_inputs.columns(2)
            fast = int(period.text_input('Short ROC', value = 11, key = 'coppock_curve_entry_fast1'))
            slow = int(offset.text_input('Long ROC', value = 14, key = 'coppock_curve_entry_slow1'))
            period = int(period.text_input('Coppock Curve Period', value = 10, key = 'coppock_curve_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'coppock_curve_entry_offset1'))
            entry_data1 = ta.coppock(data.Close, fast = fast, slow = slow, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. CC ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'coppock_curve_entry_comparator')
        
        # 3. CC ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 1', inputs2, key = 'coppock_curve_entry_input2')

        if entry_input_2 == 'Coppock Curve':
            fast, slow = entry_condition_inputs.columns(2)
            period, offset = entry_condition_inputs.columns(2)
            fast = int(period.text_input('Short ROC', value = 11, key = 'coppock_curve_entry_fast1'))
            slow = int(offset.text_input('Long ROC', value = 14, key = 'coppock_curve_entry_slow1'))
            period = int(period.text_input('Coppock Curve Period', value = 21, key = 'coppock_curve_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'coppock_curve_entry_offset1'))
            entry_data2 = ta.coppock(data.Close, fast = fast, slow = slow, length = period, offset = offset)
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass
        
        ######### CC EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. CC EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'coppock_curve_exit_input1')

        if exit_input_1 == 'Coppock Curve':
            fast, slow = exit_condition_inputs.columns(2)
            period, offset = exit_condition_inputs.columns(2)
            fast = int(period.text_input('Short ROC', value = 11, key = 'coppock_curve_exit_fast1'))
            slow = int(offset.text_input('Long ROC', value = 14, key = 'coppock_curve_exit_slow1'))
            period = int(period.text_input('Coppock Curve Period', value = 10, key = 'coppock_curve_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'coppock_curve_exit_offset1'))
            exit_data1 = ta.coppock(data.Close, fast = fast, slow = slow, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. CC EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'coppock_curve_exit_comparator')
        
        # 3. CC EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 1', inputs2, key = 'coppock_curve_exit_input2')

        if exit_input_2 == 'Coppock Curve':
            fast, slow = exit_condition_inputs.columns(2)
            period, offset = exit_condition_inputs.columns(2)
            fast = int(period.text_input('Short ROC', value = 11, key = 'coppock_curve_exit_fast1'))
            slow = int(offset.text_input('Long ROC', value = 14, key = 'coppock_curve_exit_slow1'))
            period = int(period.text_input('Coppock Curve Period', value = 21, key = 'coppock_curve_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'coppock_curve_exit_offset2'))
            exit_data2 = ta.coppock(data.Close, fast = fast, slow = slow, length = period, offset = offset)
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_positive_directional_index(num_stream, data, start_date, end_date):
    
        inputs1 = ['+DI']
        inputs2 = ['-DI', 'ADX', '+DI', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### +DI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. +DI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = '+di_entry_input1')

        if entry_input_1 == '+DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 14, key = '+di_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_entry_offset1'))
            entry_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. +DI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = '+di_entry_comparator')

        # 3. +DI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = '+di_entry_input2')

        if entry_input_2 == '+DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 21, key = '+di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'ADX':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == '-DI':
            period, offset = entry_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '+di_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_entry_offset2'))
            entry_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass
        
        ######### +DI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. +DI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = '+di_exit_input1')

        if exit_input_1 == '+DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 14, key = '+di_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_exit_offset1'))
            exit_data1 = ta.adx(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. +DI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = '+di_exit_comparator')

        # 3. +DI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = '+di_exit_input2')

        if exit_input_2 == '+DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('+DI Period', value = 21, key = '+di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'ADX':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('ADX Period', value = 14, key = 'adx_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'adx_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == '-DI':
            period, offset = exit_condition_inputs.columns(2) 
            period = int(period.text_input('-DI Period', value = 14, key = '+di_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = '+di_exit_offset2'))
            exit_data2 = ta.adx(high = data.High, low = data.Low, close = data.Close, 
                                 length = period, offset = offset).iloc[:,2]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_rsi(num_stream, data, start_date, end_date):

        inputs1 = ['RSI']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '>, Crossing Up', '==, Equal To']

        ######### RSI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. RSI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'rsi_entry_input1')

        if entry_input_1 == 'RSI':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('RSI Period', value = 14, key = 'rsi_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'rsi_entry_offset1'))
            entry_data1 = ta.rsi(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. RSI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'rsi_entry_comparator')

        # 3. RSI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'rsi_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 30, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass
        
        ######### RSI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. RSI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'rsi_exit_input1')

        if exit_input_1 == 'RSI':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('RSI Period', value = 14, key = 'rsi_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'rsi_exit_offset1'))
            exit_data1 = ta.rsi(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. RSI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'rsi_exit_comparator')

        # 3. RSI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'rsi_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 70, min_value = 0, 
                                                              max_value = 100, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_macd_signal(num_stream, data, start_date, end_date):

        inputs1 = ['MACD Signal']
        inputs2 = ['MACD', 'MACD Histogram', 'Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### MACD SIGNAL ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. MACD SIGNAL ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'macdsignal_entry_input1')

        if entry_input_1 == 'MACD Signal':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_entry_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_entry_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_entry_signal1'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdsignal_entry_offset1'))
            entry_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,2]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. MACD SIGNAL ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'macdsignal_entry_comparator')
        
        # 3. MACD SIGNAL ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'macdsignal_entry_input2')

        if entry_input_2 == 'MACD':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_entry_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_entry_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_entry_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdsignal_entry_offset2'))
            entry_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'MACD Histogram':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_entry_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_entry_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_entry_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdsignal_entry_offset2'))
            entry_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 0, min_value = 0, key = 'number1')
        else:
            pass
        
        ######### MACD SIGNAL EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('exit CONDITION', False)

        # 1. MACD SIGNAL EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'macdsignal_exit_input1')

        if exit_input_1 == 'MACD Signal':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_exit_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_exit_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_exit_signal1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdsignal_exit_offset1'))
            exit_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,2]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. MACD SIGNAL EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'macdsignal_exit_comparator')
        
        # 3. MACD SIGNAL EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'macdsignal_exit_input2')

        if exit_input_2 == 'MACD':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_exit_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_exit_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_exit_signal2'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdsignal_exit_offset2'))
            exit_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'MACD Histogram':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdsignal_exit_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdsignal_exit_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdsignal_exit_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdsignal_exit_offset2'))
            exit_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 0, min_value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_aroon_oscillator(num_stream, data, start_date, end_date):

        inputs1 = ['Aroon Oscillator']
        inputs2 = ['Number', 'Aroon Down', 'Aroon Up']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### AROON OSC ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. AROON OSC ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'aroonosc_entry_input1')

        if entry_input_1 == 'Aroon Oscillator':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Oscillator Period', value = 40, key = 'aroonosc_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_entry_offset1'))
            entry_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,2]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. AROON OSC ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'aroonosc_entry_comparator')

        # 3. AROON OSC ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'aroonosc_entry_input2')
        
        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        elif entry_input_2 == 'Aroon Up':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroonosc_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Aroon Down':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonosc_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        else:
            pass
        
        ######### AROON OSC EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. AROON OSC EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'aroonosc_exit_input1')

        if exit_input_1 == 'Aroon Oscillator':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Oscillator Period', value = 40, key = 'aroonosc_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_exit_offset1'))
            exit_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,2]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. AROON OSC EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'aroonosc_exit_comparator')

        # 3. AROON OSC EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'aroonosc_exit_input2')
        
        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        elif exit_input_2 == 'Aroon Up':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroonosc_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Aroon Down':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonosc_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonosc_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_stochrsi_fastk(num_stream, data, start_date, end_date):

        inputs1 = ['Stochastic RSI FastK']
        inputs2 = ['Number']
        mas = ['sma', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'pwma', 'rma', 'sinwma', 
               'dema', 'swma', 't3', 'tema', 'trima', 'vidya', 'wma', 'zlma']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### STOCH RSI FASTK ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. STOCH RSI FASTK ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'stochrsifk_entry_input1')

        if entry_input_1 == 'Stochastic RSI FastK':
            rsi_period, k_period = entry_condition_inputs.columns(2)
            matype, offset = entry_condition_inputs.columns(2)
            rsi_period = int(rsi_period.text_input('RSI Period', value = 14, key = 'stochrsifk_entry_rp1'))
            k_period = int(k_period.text_input('K Period', value = 3, key = 'stochrsifk_entry_kp1'))
            matype = matype.selectbox('MA Type', mas, key = 'stochrsifk_entry_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'stochrsifk_entry_offset1'))
            entry_data1 = ta.stochrsi(data.Close, rsi_length = rsi_period, k = k_period, mamode = matype, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. STOCH RSI FASTK ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'stochrsifk_entry_comparator')
        
        # 3. STOCH RSI FASTK ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'stochrsifk_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 30, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass
        
        ######### STOCH RSI FASTK EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. STOCH RSI FASTK EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'stochrsifk_exit_input1')

        if exit_input_1 == 'Stochastic RSI FastK':
            rsi_period, k_period = exit_condition_inputs.columns(2)
            matype, offset = exit_condition_inputs.columns(2)
            rsi_period = int(rsi_period.text_input('RSI Period', value = 14, key = 'stochrsifk_exit_rp1'))
            k_period = int(k_period.text_input('K Period', value = 3, key = 'stochrsifk_exit_kp1'))
            matype = matype.selectbox('MA Type', mas, key = 'stochrsifk_exit_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'stochrsifk_exit_offset1'))
            exit_data1 = ta.stochrsi(data.Close, rsi_length = rsi_period, k = k_period, mamode = matype, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. STOCH RSI FASTK EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'stochrsifk_exit_comparator')
        
        # 3. STOCH RSI FASTK EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'stochrsifk_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 70, min_value = 0, 
                                                              max_value = 100, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_stochrsi_fastd(num_stream, data, start_date, end_date):

        inputs1 = ['Stochastic RSI FastD']
        inputs2 = ['Number']
        mas = ['sma', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'pwma', 'rma', 'sinwma', 
               'dema', 'swma', 't3', 'tema', 'trima', 'vidya', 'wma', 'zlma']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### STOCH RSI FASTD ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. STOCH RSI FASTD ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'stochrsifd_entry_input1')

        if entry_input_1 == 'Stochastic RSI FastD':
            rsi_period, d_period = entry_condition_inputs.columns(2)
            matype, offset = entry_condition_inputs.columns(2)
            rsi_period = int(rsi_period.text_input('RSI Period', value = 14, key = 'stochrsifd_entry_rp1'))
            d_period = int(d_period.text_input('D Period', value = 3, key = 'stochrsifd_entry_dp1'))
            matype = matype.selectbox('MA Type', mas, key = 'stochrsifd_entry_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'stochrsifd_entry_offset1'))
            entry_data1 = ta.stochrsi(data.Close, rsi_length = rsi_period, d = d_period, mamode = matype, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. STOCH RSI FASTD ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'stochrsifd_entry_comparator')
        
        # 3. STOCH RSI FASTD ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'stochrsifd_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 30, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass
        
        ######### STOCH RSI FASTD EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. STOCH RSI FASTD EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'stochrsifd_exit_input1')

        if exit_input_1 == 'Stochastic RSI FastD':
            rsi_period, d_period = exit_condition_inputs.columns(2)
            matype, offset = exit_condition_inputs.columns(2)
            rsi_period = int(rsi_period.text_input('RSI Period', value = 14, key = 'stochrsifd_exit_rp1'))
            d_period = int(d_period.text_input('D Period', value = 3, key = 'stochrsifd_exit_dp1'))
            matype = matype.selectbox('MA Type', mas, key = 'stochrsifd_exit_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'stochrsifd_exit_offset1'))
            exit_data1 = ta.stochrsi(data.Close, rsi_length = rsi_period, d = d_period, mamode = matype, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. STOCH RSI FASTD EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'stochrsifd_exit_comparator')
        
        # 3. STOCH RSI FASTD EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'stochrsifd_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 70, min_value = 0, 
                                                              max_value = 100, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_ultimate_oscillator(num_stream, data, start_date, end_date):

        inputs1 = ['Ultimate Oscillator']
        inputs2 = ['Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### UO ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. UO ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'uo_entry_input1')

        if entry_input_1 == 'Ultimate Oscillator':
            slow, fast = entry_condition_inputs.columns(2)
            medium, offset = entry_condition_inputs.columns(2)
            slow = int(slow.text_input('Slow', value = 28, key = 'uo_entry_fast1'))
            fast = int(fast.text_input('Fast', value = 7, key = 'uo_entry_slow1'))
            medium = int(medium.text_input('Medium', value = 14, key = 'uo_entry_signal1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'uo_entry_offset1'))
            entry_data1 = ta.uo(high = data.High, low = data.Low, close = data.Close, fast = fast, slow = slow, 
                                  medium = medium, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. UO ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'uo_entry_comparator')
        
        # 3. UO ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'uo_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 50, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass

        ######### UO EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. UO EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'uo_exit_input1')

        if exit_input_1 == 'Ultimate Oscillator':
            slow, fast = exit_condition_inputs.columns(2)
            medium, offset = exit_condition_inputs.columns(2)
            slow = int(slow.text_input('Slow', value = 28, key = 'uo_exit_fast1'))
            fast = int(fast.text_input('Fast', value = 7, key = 'uo_exit_slow1'))
            medium = int(medium.text_input('Medium', value = 14, key = 'uo_exit_signal1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'uo_exit_offset1'))
            exit_data1 = ta.uo(high = data.High, low = data.Low, close = data.Close, fast = fast, slow = slow, 
                                  medium = medium, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. UO EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'uo_exit_comparator')
        
        # 3. UO EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'uo_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 50, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_aroon_up(num_stream, data, start_date, end_date):

        inputs1 = ['Aroon Up']
        inputs2 = ['Aroon Down', 'Aroon Up', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### AROONU ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. AROONU ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'aroonu_entry_input1')

        if entry_input_1 == 'Aroon Up':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonu_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_entry_offset1'))
            entry_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. AROONU ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'aroonu_entry_comparator')

        # 3. AROONU ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'aroonu_entry_input2')
        
        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number1')
        elif entry_input_2 == 'Aroon Up':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroonu_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Aroon Down':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonu_entry_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_entry_offset2'))
            entry_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        else:
            pass
        
        ######### AROONU EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. AROONU EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'aroonu_exit_input1')

        if exit_input_1 == 'Aroon Up':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonu_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_exit_offset1'))
            exit_data1 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. AROONU EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'aroonu_exit_comparator')

        # 3. AROONU EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'aroonu_exit_input2')
        
        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', min_value = 0, value = 0, key = 'number2')
        elif exit_input_2 == 'Aroon Up':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Up Period', value = 40, key = 'aroonu_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Aroon Down':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Aroon Down Period', value = 40, key = 'aroonu_exit_period2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'aroonu_exit_offset2'))
            exit_data2 = ta.aroon(data.High, data.Low, length = period, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_bollinger_bands(num_stream, data, start_date, end_date):

        inputs1 = ['Close', 'Open', 'High', 'Low']
        inputs2 = ['Lower BB', 'Upper BB', 'Middle BB']
        mas = ['sma', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'pwma', 'rma', 'sinwma', 
               'dema', 'swma', 't3', 'tema', 'trima', 'vidya', 'wma', 'zlma']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### BB ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. BB ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'bbands_entry_input1')
        
        entry_data1 = data[f'{entry_input_1}']
        entry_data1.index = entry_data1.index.astype(str)
        entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
        entry_data1.index = pd.to_datetime(entry_data1.index)

        # 2. BB ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'bbands_entry_comparator')
        
        # 3. BB ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'bbands_entry_input2')

        if entry_input_2 == 'Lower BB':
            period, std = entry_condition_inputs.columns(2)
            matype, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_entry_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_entry_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_entry_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_entry_offset1'))
            entry_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,0]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Upper BB':
            period, std = entry_condition_inputs.columns(2)
            matype, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_entry_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_entry_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_entry_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_entry_offset1'))
            entry_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,2]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        elif entry_input_2 == 'Middle BB':
            period, std = entry_condition_inputs.columns(2)
            matype, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_entry_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_entry_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_entry_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_entry_offset1'))
            entry_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data1.index)
        else:
            pass
        
        ######### BB EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. BB EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'bbands_exit_input1')
        
        exit_data1 = data[f'{exit_input_1}']
        exit_data1.index = exit_data1.index.astype(str)
        exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
        exit_data1.index = pd.to_datetime(exit_data1.index)

        # 2. BB EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'bbands_exit_comparator')
        
        # 3. BB EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'bbands_exit_input2')

        if exit_input_2 == 'Lower BB':
            period, std = exit_condition_inputs.columns(2)
            matype, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_exit_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_exit_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_exit_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_exit_offset1'))
            exit_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,0]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Upper BB':
            period, std = exit_condition_inputs.columns(2)
            matype, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_exit_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_exit_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_exit_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_exit_offset1'))
            exit_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,2]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        elif exit_input_2 == 'Middle BB':
            period, std = exit_condition_inputs.columns(2)
            matype, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 20, key = 'bbands_exit_p1'))
            std = int(std.text_input('Standard Deviation', value = 2, key = 'bbands_exit_std1'))
            matype = matype.selectbox('MA Type', mas, key = 'bbands_exit_matype1')
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'bbands_exit_offset1'))
            exit_data2 = ta.bbands(close = data.Close, period = period, std = std, 
                                    mamode = matype, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data1.index)
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_trix(num_stream, data, start_date, end_date):

        inputs1 = ['TRIX']
        inputs2 = ['Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### TRIX ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. TRIX ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'trix_entry_input1')

        if entry_input_1 == 'TRIX':
            period, signal = entry_condition_inputs.columns(2)
            scalar, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 30, key = 'trix_entry_p1'))
            signal = int(signal.text_input('Signal', value = 9, key = 'trix_entry_s1'))
            scalar = int(scalar.text_input('Scalar (Optional)', value = 100, key = 'trix_entry_sc1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'trix_entry_offset1'))
            entry_data1 = (ta.trix(close = data.Close, length = period, signal = signal, 
                                  scalar = scalar, offset = offset)*100).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. TRIX ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'trix_entry_comparator')
        
        # 3. TRIX ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'trix_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 0, key = 'number1')
        else:
            pass
        
        ######### TRIX EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. TRIX EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'trix_exit_input1')

        if exit_input_1 == 'TRIX':
            period, signal = exit_condition_inputs.columns(2)
            scalar, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('Period', value = 30, key = 'trix_exit_p1'))
            signal = int(signal.text_input('Signal', value = 9, key = 'trix_exit_s1'))
            scalar = int(scalar.text_input('Scalar (Optional)', value = 100, key = 'trix_exit_sc1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'trix_exit_offset1'))
            exit_data1 = (ta.trix(close = data.Close, length = period, signal = signal, 
                                  scalar = scalar, offset = offset)*100).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. TRIX EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'trix_exit_comparator')
        
        # 3. TRIX EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'trix_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2

def implement_cci(num_stream, data, start_date, end_date):

        inputs1 = ['CCI']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']

        ######### CCI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. CCI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'cci_entry_input1')

        if entry_input_1 == 'CCI':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('CCI Period', value = 21, key = 'cci_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cci_entry_offset1'))
            entry_data1 = ta.cci(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. CCI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'cci_entry_comparator')

        # 3. CCI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'cci_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = -100, key = 'number1')
        else:
            pass
        
        ######### CCI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. CCI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'cci_exit_input1')

        if exit_input_1 == 'CCI':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('CCI Period', value = 21, key = 'cci_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'cci_exit_offset1'))
            exit_data1 = ta.cci(high = data.High, low = data.Low, close = data.Close, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. CCI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'cci_exit_comparator')

        # 3. CCI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'cci_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 100, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_macd(num_stream, data, start_date, end_date):

        inputs1 = ['MACD']
        inputs2 = ['MACD Signal', 'MACD Histogram', 'Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### MACD ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. MACD ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'macdl_entry_input1')

        if entry_input_1 == 'MACD':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_entry_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_entry_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_entry_signal1'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdl_entry_offset1'))
            entry_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,0]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. MACD ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'macdl_entry_comparator')
        
        # 3. MACD ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'macdl_entry_input2')

        if entry_input_2 == 'MACD Signal':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_entry_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_entry_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_entry_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdl_entry_offset2'))
            entry_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,2]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'MACD Histogram':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_entry_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_entry_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_entry_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdl_entry_offset2'))
            entry_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            entry_data2.index = entry_data2.index.astype(str)
            entry_data2 = entry_data2[entry_data2.index >= str(start_date)]
            entry_data2.index = pd.to_datetime(entry_data2.index)
        elif entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 0, min_value = 0, key = 'number1')
        else:
            pass
        
        ######### MACD EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. MACD exit DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'macdl_exit_input1')

        if exit_input_1 == 'MACD':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_exit_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_exit_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_exit_signal1'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdl_exit_offset1'))
            exit_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,0]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. MACD EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'macdl_exit_comparator')
        
        # 3. MACD EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'macdl_exit_input2')

        if exit_input_2 == 'MACD Signal':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_exit_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_exit_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_exit_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdl_exit_offset2'))
            exit_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,2]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'MACD Histogram':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdl_exit_fast2'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdl_exit_slow2'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdl_exit_signal2'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'macdl_exit_offset2'))
            exit_data2 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            exit_data2.index = exit_data2.index.astype(str)
            exit_data2 = exit_data2[exit_data2.index >= str(start_date)]
            exit_data2.index = pd.to_datetime(exit_data2.index)
        elif exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 0, min_value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_macd_histogram(num_stream, data, start_date, end_date):

        inputs1 = ['MACD Histogram']
        inputs2 = ['Number']
        entry_conditions = ['>, Crossing Up', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['<, Crossing Down', '>, Crossing Up', '==, Equal To']

        ######### MACD HISTOGRAM ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. MACD HISTOGRAM ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'macdhist_entry_input1')

        if entry_input_1 == 'MACD Histogram':
            fast_ma, slow_ma = entry_condition_inputs.columns(2)
            signal_period, offset = entry_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdhist_entry_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdhist_entry_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdhist_entry_signal1'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdhist_entry_offset1'))
            entry_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. MACD HISTOGRAM ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'macdhist_entry_comparator')
        
        # 3. MACD HISTOGRAM ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'macdhist_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 0, key = 'number1')
        else:
            pass
        
        ######### MACD HISTOGRAM EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)

        # 1. MACD HISTOGRAM EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'macdhist_exit_input1')

        if exit_input_1 == 'MACD Histogram':
            fast_ma, slow_ma = exit_condition_inputs.columns(2)
            signal_period, offset = exit_condition_inputs.columns(2)
            fast_ma = int(fast_ma.text_input('Fast MA', value = 12, key = 'macdhist_exit_fast1'))
            slow_ma = int(slow_ma.text_input('Slow MA', value = 26, key = 'macdhist_exit_slow1'))
            signal_period = int(signal_period.text_input('Signal Period', value = 9, key = 'macdhist_exit_signal1'))
            offset = int(offset.text_input('Offset', value = 0, key = 'macdhist_exit_offset1'))
            exit_data1 = ta.macd(high = data.High, low = data.Low, close = data.Close, fast = fast_ma, slow = slow_ma, 
                                  signal = signal_period, offset = offset).iloc[:,1]
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. MACD HISTOGRAM EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'macdhist_exit_comparator')
        
        # 3. MACD HISTOGRAM EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'macdhist_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 0, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
        
def implement_mfi(num_stream, data, start_date, end_date):

        inputs1 = ['MFI']
        inputs2 = ['Number']
        entry_conditions = ['<, Crossing Down', '<, Crossing Down', '==, Equal To']
        exit_conditions = ['>, Crossing Up', '>, Crossing Up', '==, Equal To']

        ######### MFI ENTRY CONDITION #########

        num_stream.sidebar.markdown('')
        entry_condition_inputs = num_stream.sidebar.expander('ENTRY CONDITION', False)

        # 1. MFI ENTRY DATA 1

        entry_input_1 = entry_condition_inputs.selectbox('Input 1', inputs1, key = 'mfi_entry_input1')

        if entry_input_1 == 'MFI':
            period, offset = entry_condition_inputs.columns(2)
            period = int(period.text_input('MFI Period', value = 14, key = 'mfi_entry_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mfi_entry_offset1'))
            entry_data1 = ta.mfi(high = data.High, low = data.Low, close = data.Close, 
                                 volume = data.Volume, length = period, offset = offset)
            entry_data1.index = entry_data1.index.astype(str)
            entry_data1 = entry_data1[entry_data1.index >= str(start_date)]
            entry_data1.index = pd.to_datetime(entry_data1.index)
        else:
            pass

        # 2. MFI ENTRY COMPARATOR

        entry_comparator = entry_condition_inputs.selectbox('Comparator', entry_conditions, key = 'mfi_entry_comparator')

        # 3. MFI ENTRY DATA 2

        entry_input_2 = entry_condition_inputs.selectbox('Input 2', inputs2, key = 'mfi_entry_input2')

        if entry_input_2 == 'Number':
            entry_data2 = entry_condition_inputs.number_input('Specify Input Value', value = 30, min_value = 0, 
                                                              max_value = 100, key = 'number1')
        else:
            pass
        
        ######### MFI EXIT CONDITION #########

        num_stream.sidebar.markdown('')
        exit_condition_inputs = num_stream.sidebar.expander('exit CONDITION', False)

        # 1. MFI EXIT DATA 1

        exit_input_1 = exit_condition_inputs.selectbox('Input 1', inputs1, key = 'mfi_exit_input1')

        if exit_input_1 == 'MFI':
            period, offset = exit_condition_inputs.columns(2)
            period = int(period.text_input('MFI Period', value = 14, key = 'mfi_exit_period1'))
            offset = int(offset.text_input('Offset (Optional)', value = 0, key = 'mfi_exit_offset1'))
            exit_data1 = ta.mfi(high = data.High, low = data.Low, close = data.Close, 
                                 volume = data.Volume, length = period, offset = offset)
            exit_data1.index = exit_data1.index.astype(str)
            exit_data1 = exit_data1[exit_data1.index >= str(start_date)]
            exit_data1.index = pd.to_datetime(exit_data1.index)
        else:
            pass

        # 2. MFI EXIT COMPARATOR

        exit_comparator = exit_condition_inputs.selectbox('Comparator', exit_conditions, key = 'mfi_exit_comparator')

        # 3. MFI EXIT DATA 2

        exit_input_2 = exit_condition_inputs.selectbox('Input 2', inputs2, key = 'mfi_exit_input2')

        if exit_input_2 == 'Number':
            exit_data2 = exit_condition_inputs.number_input('Specify Input Value', value = 70, min_value = 0, 
                                                              max_value = 100, key = 'number2')
        else:
            pass
        
        return entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2
    
def crossingdown_crossingup(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)

    else:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

def crossingdown_crossingdown(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

def crossingdown_equalto(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)

    else:
        for i in range(len(prices)):
            if entry_data1[i] < entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)

    return buy_price, sell_price, strategy_signals

def crossingup_crossingdown(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)

    return buy_price, sell_price, strategy_signals

def crossingup_crossingup(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

def crossingup_equalto(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] > entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)

    return buy_price, sell_price, strategy_signals

def equalto_crossingup(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] > exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

def equalto_crossingdown(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] < exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

def equalto_equalto(data, entry_data1, entry_data2, exit_data1, exit_data2):
    buy_price = []
    sell_price = []
    strategy_signals = []
    prices = data.Close
    signal = 0
    
    if type(entry_data2) == float or type(exit_data2) == float or type(entry_data2) == int or type(exit_data2) == int:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
    
    else:
        for i in range(len(prices)):
            if entry_data1[i] == entry_data2[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)

            elif exit_data1[i] == exit_data2[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    strategy_signals.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    strategy_signals.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                strategy_signals.append(0)
                        
    return buy_price, sell_price, strategy_signals

