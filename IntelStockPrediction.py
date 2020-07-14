# Importing the libraries

# ================================= 
# MOST IMPORTANT library
# =================================
import tensorflow
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from pathlib import Path

# ================================= 
# Other libraries just as important
# =================================

import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json
import os
import quandl
import datetime

#=================================================================================
#               Processing the data from API based on User Input and preference
#================================================================================

# type the name
print("Example: Google: GOOGL | Intel: INTL ")
print("---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ")
print("---- ---- ---- ---- ---- ---- ---- ---- GET RICH - 2020 ---- ---- ---- ---- ---- ---- ---- ---- ")
print("---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ")
ticker = input("Enter the ticker of a company (Search name of company for their ticker)? ")
type(ticker)

alpha_api_key = 'C5C28C1UTUGHH30R'


#file_path = 'api.txt'
#contents = Path(file_path).read_text()
#alpha_api_key = contents



# JSON file with all the stock market data for AAL from the last 20 years
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker, alpha_api_key)

# Save data to two files

file_to_save = 'C:/temp/StockFile-Train-%s.csv'%ticker    # this file would be our training file
test_file = 'C:/temp/StockFile-Test-%s.csv'%ticker        # this file would be our test file

if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),float(v['4. close']),float(v['1. open']),float(v['5. volume'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
    print('Data saved to : %s'%file_to_save)
    df.to_csv(file_to_save)
    print('Data saved to : %s'%test_file)
    df.to_csv(test_file)

else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save, index_col=0)
    df = pd.read_csv(test_file, index_col=0)

df = df.sort_values('Date')
df.head()


# Importing the training set
dataset_train = pd.read_csv(file_to_save, index_col=0)
#dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#=================================================================================
#               BUILDING THE NEURAL NETWORK
#=================================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# adding an LSTM layer with 50 neurons, stacked with another
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
# the rate of neurons we want to ignore, hence the name of dropping, during the iteration of training
regressor.add(Dropout(0.2))

# adding another LSTM layer with 50 neurons, stacked with the first
regressor.add(LSTM(units = 50, return_sequences = True))
# the rate of neurons we want to ignore, hence the name of dropping, during the iteration of training
regressor.add(Dropout(0.2))

# adding a third LSTM layer with 50 neurons, stacked with the second
regressor.add(LSTM(units = 50, return_sequences = True))
# the rate of neurons we want to ignore, hence the name of dropping, during the iteration of training
regressor.add(Dropout(0.2))

# adding a fourth LSTM layer with 50 neurons, stacked with the third and is the last one
regressor.add(LSTM(units = 50))
# the rate of neurons we want to ignore, hence the name of dropping, during the iteration of training
regressor.add(Dropout(0.2))

# adding the output layer of our neural network
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
# dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

dataset_test = pd.read_csv(test_file, index_col=0)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real %s Stock Price'%ticker)
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted %s Stock Price'%ticker)
plt.title('%s Stock Price Prediction'%ticker)
plt.xlabel('Time')
plt.ylabel('%s Stock Price'%ticker)
plt.legend()
plt.show()