# **MOD007893 TRI2 F01CAM : ASSIGNMENT 010**

# **Section 2: Impact Analysis of Renewable Energy Adoption in the UK to Achieve Net-Zero Goals by 2050: Model Implmentation **

#  **SID : 2158452**
#  **GROUP : TECHIE PENTA**

import pandas as pd
import numpy as np
net_zero =pd.read_csv('net_zero.csv' ,index_col=0)

#Convert index to datetime
net_zero.index=pd.to_datetime(net_zero.index)

#Create new variables
net_zero =(net_zero
     #A lag variable
     .assign(Emission_lag = lambda df: df.Emissions.shift(1))
     .assign(Day = lambda df: df.index.dayofweek)
     .assign(Month= lambda df: df.index.month)
     .assign(Year =lambda df: df.index.year)
    # A moving average variable
     .assign(Emissions_ma =lambda df: df.Emissions.rolling(window=3).mean())
     # drop the first two variables
     .dropna(subset=['Emissions_ma'], axis=0)

)
print(net_zero.head())

### Split the dataset into training and  test sets based on the timestamps

# Set cutoff date for train/test split (The last day of 2018)
cutoff_date = "2019-06-30"

# Split the data into training and testing sets
train_data = net_zero.loc[net_zero.index < cutoff_date]
test_data = net_zero.loc[net_zero.index >= cutoff_date]

# Verify the shapes of the training and testing sets
print('Training data shape:', train_data.shape)
print('Testing data shape:', test_data.shape)

#Scale the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# print out the adjustment the scaler applied to the Emissions column
emission_index = 6
print(f'Emissions values were scaled by multiplying {scaler.scale_[emission_index]:.10f} and adding {scaler.min_[emission_index]:.6f}')

scale = scaler.scale_[emission_index]
min_value = scaler.min_[emission_index]
print(f'The scale for emission is {scale}')
print(f'The minimum value for scaling emission is {min_value}')

#import supplementary libaries
import datetime
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# Set up the TensorBoard callback
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Extract the input features (X) and target variable (y) for the training set
X_train = scaled_train_data[:, :-1]
y_train = scaled_train_data[:,-1]

# Extract the input features (X) and target variable (y) for the testing set
X_test = scaled_test_data[:, :-1]
y_test = scaled_test_data[:,-1]

# Reshape input features to be 3D for the LSTM layer
X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test = X_test.reshape(-1, 1, X_test.shape[1])

# Define model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu'))
model.add(Dense(50, input_dim=6, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model with the TensorBoard callback
model.fit(X_train, y_train, epochs=50, shuffle=True, verbose=2, callbacks=[tensorboard_callback])

# Make predictions on test set
y_pred = model.predict(X_test)

# Flatten the prediction array and convert it to a series
y_pred =pd.Series(y_pred.flatten())

#inverse transform back to the original value
y_inv =(y_pred-min_value)/scale

#build a comparison dataframe to compare the actual values with the predicted
y_actual = test_data.reset_index().loc[:,'Emissions']
eval =(pd.concat([y_actual, y_inv], axis=1)
        .rename(columns={'Emissions': 'Actual', 0: 'Predicted'})
)
print(eval.head())


#Calculate error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_actual, y_inv)
mse = mean_squared_error(y_actual, y_inv)
rmse = np.sqrt(mse)

# Print error metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Save the model to an HDF5 file
model.save('net_zero.h5')
