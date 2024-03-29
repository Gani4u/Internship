import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

# FOR REPRODUCIBILITY

np.random.seed(7)


# IMPORTING DATASET 

dataset = pd.read_csv('EOD-AAPL_in.csv', usecols=[1,2,3,4])

dataset = dataset.reindex(index = dataset.index[::-1])


# CREATING OWN INDEX FOR FLEXIBILITY

obs = np.arange(1, len(dataset) + 1, 1)


# TAKING DIFFERENT INDICATORS FOR PREDICTION

OHLC_avg = dataset.mean(axis = 1) #calculates the mean of each row in the dataset

HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)

close_val = dataset[['Close']]



# PLOTTING ALL INDICATORS IN ONE PLOT

plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')

plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')

plt.plot(obs, close_val, 'g', label = 'Closing price')

plt.legend(loc = 'upper right')

plt.show()



# PREPARATION OF TIME SERIES DATASET

OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1))

scaler = MinMaxScaler(feature_range=(0, 1))

OHLC_avg = scaler.fit_transform(OHLC_avg)


# TRAIN-TEST SPLIT

train_OHLC = int(len(OHLC_avg) * 0.75)

test_OHLC = len(OHLC_avg) - train_OHLC

train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]



# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
# (which is suitable for training a time series prediction model like an LSTM)

def new_dataset(dataset, step_size): #step_size :It determines how many previous data points will be used to predict the next data point.

	data_X, data_Y = [], []

	for i in range(len(dataset)-step_size-1):

		a = dataset[i:(i+step_size), 0]

		data_X.append(a)

		data_Y.append(dataset[i + step_size, 0])

	return np.array(data_X), np.array(data_Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY



# TIME-SERIES DATASET

trainX, trainY = new_dataset(train_OHLC, 1);

testX, testY = new_dataset(test_OHLC, 1);


# RESHAPING TRAIN AND TEST DATA
# reshaped to match the input requirements of the LSTM model.
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

step_size = 1;   




# LSTM MODEL

model = Sequential()

model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))

model.add(LSTM(16))

model.add(Dense(1))

model.add(Activation('linear'))



# MODEL COMPILING AND TRAINING

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

'''
The number of epochs (iterations over the entire training dataset) to train the model. 
In this case, it is set to 10.

batch_size=1:  meaning the model will update its weights after each individual sample.
'''

'''
verbose = 0: No logging output is shown during training.
verbose = 1: Display a progress bar for each epoch, showing the number of the current epoch and the training loss.
verbose = 2: Display a progress bar for each epoch, but only show one line per epoch, indicating the completion of each epoch.
'''


# PREDICTION

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

for i in range(10):
    print(testPredict[i],testY[i])

plt.figure(figsize=(16,8))
plt.plot(testY,color='red',label='test')
plt.plot(testPredict,color='green',label='pred')
plt.legend()
plt.show()



# DE-NORMALIZING FOR PLOTTING

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])


# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS

trainPredictPlot = np.empty_like(OHLC_avg) # Creates an array that has same shape as OHLC_avg, but will have random vales

trainPredictPlot[:, :] = np.nan

trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict



# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS

testPredictPlot = np.empty_like(OHLC_avg)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict



# DE-NORMALIZING MAIN DATASET 

OHLC_avg = scaler.inverse_transform(OHLC_avg)



# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS

plt.plot(OHLC_avg, 'g', label = 'original dataset')

plt.plot(trainPredictPlot, 'r', label = 'training set')

plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')

plt.legend(loc = 'upper right')

plt.xlabel('Time in Days')

plt.ylabel('OHLC Value of Apple Stocks')

plt.show()