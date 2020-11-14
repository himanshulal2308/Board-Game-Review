import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('games.csv')

#Remove any rows without user reviews
dataset = dataset[dataset["users_rated"]>0]

#Remove any rows with any missing values
dataset = dataset.dropna(axis=0)

plt.hist(dataset["average_rating"])
plt.show()

corrmat = dataset.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()

#get all columns from data frame
columns = dataset.columns.tolist()

#filter the columns to remove data we don't want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#store the variable we'll be predicting on 
target="average_rating"

from sklearn.model_selection import train_test_split
train = dataset.sample(frac=0.8,random_state=1)
test = dataset.loc[~dataset.index.isin(train.index)]
print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LR = LinearRegression()
LR.fit(train[columns],train[target])
predictions = LR.predict(test[columns])

#compute error between actual values and predicted values
print(mean_squared_error(predictions,test[target]))


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
RFR.fit(train[columns],train[target])
predictionss = RFR.predict(test[columns])
print(mean_squared_error(predictionss,test[target]))
