# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.kersa.callbacks import EarlyStopping, ModelCheckpoint

# %% [markdown]
# import data set

# %%
data_set = pd.read_csv("./data/train.csv")
data_set.head(10)

# %% [markdown]
# check for null value and remove them or fill them
# 

# %%
data_set.info()

# %% [markdown]
# above is retrieving the data types

# %% [markdown]
# dropping unneccessary columns eg store  and items 

# %%
data_set = data_set.drop(['store', 'item'], axis=1)

# %%
# getting the new data set
data_set.info()

# %% [markdown]
# convert date datatype from object to datetime datatype

# %%
data_set['date'] = pd.to_datetime(data_set['date'])

# %%
#check the datatype
data_set.info()

# %% [markdown]
# converting date to a month period and sum the number of items in each month

# %%
data_set['date'] = data_set['date'].dt.to_period('M')
monthly_sales = data_set.groupby('date').sum().reset_index()

# %% [markdown]
# convert the resulting date column to timestamp datatpe

# %%
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

# %%
#get the sample
monthly_sales.head(10)

# %% [markdown]
# visualize the sales for monthly items

# %%
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly Customer Sales")
plt.show()

# %% [markdown]
# call the diff between sales column to make the sales data stationary
# 

# %%
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)

# %%
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly Customer Sales Diff")
plt.show()

# %% [markdown]
# Drop date and sales column and create supervised data

# %%
supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)


# %% [markdown]
# prepare the supervised data

# %%
#using te prev 12 months sales as the input feature and te next month sale as the output
for i in range(1, 13):
    col_name = 'month' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head(10)

# %% [markdown]
# Split the data into train and test data

# %%
#this is for the prev 12 months
train_data = supervised_data[:-12]
#this is for the upcoming 12 months
test_data = supervised_data[-12:]
print("Train Data Shape: ", train_data.shape)
print("Test Data Shape: ", test_data.shape)

# %% [markdown]
# Restrict the feature values in range of  from -1 & 1

# %%
scaler = MinMaxScaler(feature_range=(-1,1))
#fit the trainer data 
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


# %%
x_train, y_train = train_data[:,1:], train_data[:,0:1]
x_test, y_test = test_data[:,1:], test_data[:,0:1] 
y_train = y_train.ravel()
y_test = y_test.ravel()
print("x-train shape: ", x_train.shape)
print("y-train shape: ", y_train.shape)
print("x-test shape: ", x_test.shape)
print("y-test shape: ", y_test.shape)

# %% [markdown]
# 1st Columns are Outputs any column after is the input

# %% [markdown]
# MAKE PREDICTION DATA FRAME TO MERGE THE PREDICTED SALES PRICES OF ALL TRAINED ALGS

# %%
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_data_frame = pd.DataFrame(sales_dates)

# %% [markdown]
# actual monthly  sales for the last 13 months

# %%
act_sales = monthly_sales['sales'][-13:].to_list()
print(act_sales)

# %% [markdown]
# CREATE THE LINEAR REGRESSION MODEL AND PREDICT OUTPUT

# %%
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train, y_train)
lr_pre = linear_reg_model.predict(x_test)

# %%
lr_pre = lr_pre.reshape(-1,1)
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

# %%
result_list = []
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_data_frame = predict_data_frame.merge(lr_pre_series, left_index= True, right_index= True)


# %%
lr_mnsd_err = np.sqrt(mean_squared_error(predict_data_frame["Linear Prediction"], monthly_sales["sales"][-12:]))
lr_abso_err = mean_absolute_error(predict_data_frame["Linear Prediction"], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_data_frame['Linear Prediction'], monthly_sales['sales'][-12:])
print("Linear Regression Mean Squared Error: ", lr_mnsd_err)
print("Linear Regression Mean Absolute Error: ", lr_abso_err)
print("Linear Regression R2 Score: ", lr_r2)


# %% [markdown]
# Visualize Actial Sales Against Predictions

# %%
plt.figure(figsize=(15,5))
#Actual sale
plt.plot(monthly_sales['date'], monthly_sales['sales'])
#Predicted sales
plt.plot(predict_data_frame['date'], predict_data_frame['Linear Prediction'])
plt.title("Customer sales forecast using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Actual Sales', 'Predicted Sales'])
plt.show()


