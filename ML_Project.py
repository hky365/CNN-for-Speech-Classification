#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import math
import os
import pprint
import tensorflow as tf


# In[2]:


data_feat = np.load('feat.npy', allow_pickle = True)
data_path = np.load('path.npy')
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
words_train = train_csv['word']
df = {"path": data_path, "Features": data_feat}


# In[3]:


## converting the df dict to pandas df
df_q = pd.DataFrame(df)

#merging the data set for train and test

train_df_tmp = pd.merge(left = train_csv, right = df_q, on = 'path')
test_df = pd.merge(left = test_csv, right = df_q, on = 'path')

#dropping the word column
train_df = train_df_tmp.drop(['word'], axis = 1)


# In[4]:


### putting the features seperately in df for train and test

## for train ...

feat_dict = {}

for i in range(len(train_df)):
    feat_list = []
    for j in range(len(train_df.iloc[i,1])):
        feat_list.append(train_df.iloc[i,1][j])
    
    feat_dict[train_df.iloc[i,0]] = feat_list


# In[5]:


## for test ...

test_dict = {}

for i in range(len(test_df)):
    test_list = []
    for j in range(len(test_df.iloc[i,1])):
        test_list.append(test_df.iloc[i,1][j])
    
    test_dict[test_df.iloc[i,0]] = test_list
    


# In[6]:


dftest_final = pd.DataFrame.from_dict(test_dict, orient = 'index')
dftrain_final = pd.DataFrame.from_dict(feat_dict, orient = 'index')


# In[7]:


##### converting words for y to integers

y_train = train_csv
from sklearn.preprocessing import LabelEncoder
from keras import utils

lb_make = LabelEncoder()
y_train["word_code"] = lb_make.fit_transform(y_train["word"])

y_trn = y_train["word_code"]
y_cat = to_categorical(y_trn)


# In[8]:


## Converting the df to array to get the length and adjust the zero matrix
atem = np.array(dftrain_final)


zero_mat = np.zeros((len(y_trn),len(atem[1]), len(atem[0][1])))


final_mat = zero_mat


# In[9]:


dftrain_final_copy = dftrain_final
dftrain_final_copy = dftrain_final_copy.replace(None, method ='pad')


# In[10]:


for i in range(len(dftrain_final_copy)):
    for j in range(len(dftrain_final_copy.iloc[0,:])):
      
        for k in range(len(dftrain_final_copy.iloc[0,1])):
            if np.isnan(dftrain_final_copy.iloc[i,j]).any():
                final_mat[i][j][k]=np.nan
            else:
                final_mat[i][j][k]=dftrain_final_copy.iloc[i,j][k]


# In[11]:


######### for splitting #####################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler



z = final_mat.reshape(final_mat.shape[0], 1287)


X_train, X_test, y_train, y_test = train_test_split(z,y_cat, test_size=0.1, random_state=25) #stratify = y_trn)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = np.reshape(X_train, (len(X_train), 99,13,1))
X_test = np.reshape(X_test, (len(X_test), 99,13,1))


# In[27]:


################### the model ####################

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import keras.optimizers
from keras import initializers




model = Sequential()
model.add(Conv2D(128, (3, 3), padding='same', 
                 input_shape=(99,13,1)))
model.add(Activation('relu'))


model.add(Conv2D(128, (3, 3),strides = 2,))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Conv2D(128, (3, 3),strides = 2,))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(35, activation='softmax'))


# In[28]:


model.compile(keras.optimizers.adam(lr=0.0003) ,loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train, y_train, validation_split = 0.1, epochs= 80)


# In[ ]:


##################ONLY RUN THIS CODE AFTER YOU ARE SATISFIED WITH YOUR MODEL###########################


# In[18]:


atem = np.array(dftest_final)

zero_mat = np.zeros((len(dftest_final),len(atem[1]), len(atem[0][1])))

test_mat = zero_mat 


# In[19]:


dftest_final_copy = dftest_final
dftest_final_copy = dftest_final_copy.replace(None, method ='pad')


# In[20]:


for i in range(len(dftest_final_copy)):
    for j in range(len(dftest_final_copy.iloc[0,:])):
      
        for k in range(len(dftest_final_copy.iloc[0,1])):
            if np.isnan(dftest_final_copy.iloc[i,j]).any():
                test_mat[i][j][k]=np.nan
            else:
                test_mat[i][j][k]=dftest_final_copy.iloc[i,j][k]


# In[21]:



test_mat = test_mat.reshape(test_mat.shape[0], 1287)

sc = StandardScaler()
test_mat = sc.fit_transform(test_mat)


# In[22]:


test_mat = test_mat.reshape(11005,99,13,1)


# In[23]:


y_test1 = model.predict_classes(test_mat)
results = lb_make.inverse_transform(y_test1)


# In[24]:


test_csv = pd.read_csv('test.csv')
final = pd.DataFrame()
final['path'] = test_csv["path"]
final["word"] = results
display(final)


# In[26]:


final.to_csv("result.csv", index = False)

