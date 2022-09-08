import numpy as np
import pandas as pd
from app.scaling import Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
import datetime as dt

pd.set_option("display.precision", 8)

class Prediction(Preprocessing):
    
    def __init__(self, exchange, interval, asset, action_model, price_model, market = None):
        super().__init__(exchange, interval, asset, market)

        self.start_date = self.df.index[-3]
        self.action_model = action_model
        self.price_model = price_model

        features = ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', 
                    'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'RSI', 'SR_K', 'SR_D', 'HL_PCT', 'PCT_CHG']
        price_features = ['High', 'Low', 'Open', 'Volume', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', 
                    'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'RSI', 'SR_K', 'SR_D', 'HL_PCT', 'PCT_CHG']
        self.df_action = self.df.copy()[features + ['Distinct_Action']]
        self.df_price = self.df.copy()[features]

        self.scaler = StandardScaler()
        self.df_price['Adj Close_Scaled'] = self.scaler.fit_transform(self.df_price[['Adj Close']].values).reshape(-1)
        self.action_features, self.action_labels = super(Prediction, self).scaling(self.df_action)
        self.price_features, self.price_labels = super(Prediction, self).scaling(self.df_price[price_features + ['Adj Close_Scaled']])
        self.ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False, handle_unknown = 'ignore')
        self.action_labels = self.ohe.fit_transform(self.action_labels)
        
    def get_prediction(self):         
        self.model_prediction_action = self.action_model.predict(self.action_features)
        self.model_prediction_price = self.price_model.predict(self.price_features)
        
        self.model_prediction_action = np.array(self.ohe.inverse_transform(self.model_prediction_action.round())).flatten()
        self.model_prediction_price = self.scaler.inverse_transform(self.model_prediction_price).flatten()
        self.requested_prediction_action = str(self.model_prediction_action[-1])
        self.requested_prediction_price = round(float(self.model_prediction_price[-1]), 8)

        price_prediction_length = self.model_prediction_price.shape[0]
        self.df_price = self.df_price.iloc[-price_prediction_length:]

        self.score_action = self.action_model.evaluate(self.action_features, self.action_labels, verbose = 0) 
        self.score_price = r2_score(self.df_price['Adj Close'].values[1:], self.model_prediction_price[:-1])
        self.score_action, self.score_price = round((self.score_action[1] * 100), 1), round((self.score_price * 100), 1)
        
    def prediction_postprocessing(self, indication):
        self.indication = indication
        indicators = {'Analysed':'Distinct_Action', 'Predicted':'Action_Predictions'}

        action_prediction_length = self.model_prediction_action.shape[0]
        self.df_visulization = self.df.iloc[-action_prediction_length:]
        self.df_visulization['Action_Predictions'] = self.model_prediction_action
        self.df_visulization = self.df_visulization[['Open', 'Adj Close', 'Volume', 'Distinct_Action', 
                                                     'Action_Predictions', 'Future_Adj_Close']]
        self.df_visulization['Price_Buy'] = self.df_visulization[self.df_visulization[indicators[self.indication]] == 'Buy']['Adj Close']
        self.df_visulization['Price_Sell'] = self.df_visulization[self.df_visulization[indicators[self.indication]] == 'Sell']['Adj Close']
        self.df_visulization['Bullish Volume'] = self.df_visulization[self.df_visulization['Adj Close'] >= self.df_visulization['Open']]['Volume']
        self.df_visulization['Bearish Volume'] = self.df_visulization[self.df_visulization['Adj Close'] < self.df_visulization['Open']]['Volume']
