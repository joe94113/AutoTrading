from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
# import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Action:
    BUY = 1
    HOLD = 0
    SELL = -1

class Trader:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 50
        self.time_frame = 20
        self.dimension = 4
        self.output = []
        self.predPrices = []

    def train(self, training_data):
        foxconndf_norm = self.normalize(training_data)
        X_train, y_train, X_test, y_test = self.data_helper(foxconndf_norm, self.time_frame)
        model = self.build_model( self.time_frame, self.dimension )
        model.fit( X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1, verbose=1)

        # 查看模型訓練結果
        pred = model.predict(X_test)
        denorm_pred = self.denormalize(training_data, pred)
        denorm_ytest = self.denormalize(training_data, y_test)
        self.draw_predict_result(denorm_pred, denorm_ytest)

        return model

    # 載入資料
    def load_data(self, df_name):
        foxconndf= pd.read_csv(df_name, names=["open", "high", "low", "close"] )
        foxconndf.dropna(how='any',inplace=True)
        return foxconndf

    # 資料正規化
    def normalize(self, df):
        newdf= df.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        newdf['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        newdf['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
        newdf['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
        newdf['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
        
        return newdf

    # 資料處理
    def data_helper(self, df, time_frame):
    
        # 資料維度: 開盤價、最高價、最低價、收盤價, 4維
        number_features = len(df.columns)

        # 將dataframe 轉成 numpy array
        datavalue = df.values
        result = []
        # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
        for index in range( len(datavalue) - (time_frame+1) ): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
            result.append(datavalue[index: index + (time_frame+1) ]) # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
        
        result = np.array(result)
        number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
        
        x_train = result[:int(number_train), :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
        y_train = result[:int(number_train), -1][:,0] # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
        
        # 測試資料
        x_test = result[int(number_train):, :-1]
        y_test = result[int(number_train):, -1][:,0]
        
        # 將資料組成變好看一點
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  

        return [x_train, y_train, x_test, y_test]
    
    # 建立模型，256個神經元的LSTM層，加上了Dropout層來防止資料過度擬合，最後兩層是全連接層
    def build_model(self, input_length, input_dim):
        d = 0.3
        model = Sequential()

        model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
        model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

        model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

        return model

    # 還原資料
    def denormalize(self, df, norm_value):
        original_value = df['open'].values.reshape(-1,1)
        norm_value = norm_value.reshape(-1,1)
        
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit_transform(original_value)
        denorm_value = min_max_scaler.inverse_transform(norm_value)
        
        return denorm_value

    # 畫出預測及真實結果
    def draw_predict_result(self, predict_value, true_value):
        plt.plot(predict_value,color='red', label='Prediction')
        plt.plot(true_value,color='blue', label='Answer')
        plt.legend(loc='best')
        plt.show()
    
    # 將新資料寫回
    def add_row_to_df(self, row, df):
        df.loc[df.index.max()+1] = [row[0],row[1],row[2],row[3]]
        return df

    # 決定是否要買進
    def predict_action(self, forecast_price, denorm_pred):
            if (len(self.output) == 0):
                print(denorm_pred[-2][0])
                if(forecast_price > denorm_pred[-2][0]):
                    action = Action.BUY
                else:
                    action = Action.SELL
            # 當持有股票，未來大於現在，則持有
            elif (self.predPrices[len(self.predPrices) -1] < forecast_price and sum(self.output) == 1):
                action = Action.HOLD
            # 當持有股票，未來小於現在，則賣出
            elif (self.predPrices[len(self.predPrices) -1] > forecast_price and sum(self.output) == 1):
                action = self.check_action(Action.SELL)
            # 當未持有股票，未來大於現在，則買入
            elif (self.predPrices[len(self.predPrices) -1] < forecast_price and sum(self.output) == 0):
                action = self.check_action(Action.BUY)
            # 當未持有股票，未來小於現在，則放空
            elif (self.predPrices[len(self.predPrices) -1] > forecast_price and sum(self.output) == 0):
                action = self.check_action(Action.SELL)
            # 放空股票時，未來大於現在，則買入
            elif (self.predPrices[len(self.predPrices) -1] < forecast_price and sum(self.output) == -1):
                action = self.check_action(Action.BUY)
            # 放空股票時，未來小於現在，則持續放空
            elif (self.predPrices[len(self.predPrices) -1] > forecast_price and sum(self.output) == -1):
                action = Action.HOLD
            else:
                action = Action.HOLD
            self.output.append(action)
            self.predPrices.append(forecast_price)
            return action
    
    # 確保沒有連續買賣
    def check_action(self, action):
        if sum(self.output) + action != 2 or sum(self.output) + action != -2:
            return action
        else:
            return Action.HOLD

