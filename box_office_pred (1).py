# -*- coding: utf-8 -*-

#import keras, scikit learn and pandas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing as pr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

#import the data preProcessed in R as a pandas data frame
fd_tr=pd.read_csv("final_tr_data.csv")
fd_te=pd.read_csv("final_te_data.csv")


#use Pandas data frame manipulation, to make the data suitable for a machine learning algorithm
x_tr=fd_tr.iloc[:,:33]
y_tr=fd_tr.iloc[:,33]

x_te=fd_te

#Cobmine the test and training data so they can both be scaled and encoded
full_x=x_tr.append(x_te)

scaler = StandardScaler()

#Scale the coninous vairables for use in the Neural Net
scale_x=scaler.fit_transform(full_x.iloc[:3000,[0,2,3,6]])
scale_x_te=scaler.transform(full_x.iloc[3000:,[0,2,3,6]])

full_x.iloc[:3000,[0,2,3,6]]=scale_x
full_x.iloc[3000:,[0,2,3,6]]= scale_x_te

# One hot encode the factor variables
full_x=pd.get_dummies(full_x)
x_tr=full_x.iloc[0:3000,:].values
x_te=full_x.iloc[3000:,:].values

#The neural net to be strongly overfitting so we have used both dropout and regularization, the value
# we use for the drop out and regulurization are just based off trial an error

y_tr=y_tr.values

model = Sequential()
model.add(Dropout(0.35))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.35))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.35))
model.add(Dense(1))
adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, 
              loss='mean_squared_error')
model.fit(x_tr, y_tr,
          batch_size=24,
          epochs=50,
          validation_split = 0.2)

pred=model.predict(x_te)

np.savetxt("nn5.csv",pred,delimiter=",")

#We also implement a Random Forest for prediction, with gird search to find the optimal hyperparameters
parameters={"max_features":[5,10,25,30],"min_samples_split":[2,3,4,5]}
clf = RandomForestRegressor(n_estimators=500, random_state=0)
clf.cv = GridSearchCV(clf, parameters, cv=3)
clf.cv.fit(x_tr,y_tr)
pred1=clf.cv.predict(x_te)

np.savetxt("rfsk1.csv",pred1,delimiter=",")






