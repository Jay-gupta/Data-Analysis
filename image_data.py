#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('pylab', 'inline')
import os
import random
from scipy.misc import imread


# In[5]:


root_dir = os.path.abspath('.')
data_dir = 'C:/Users/hkc03/Desktop/JAY'

train = pd.read_csv(os.path.join(data_dir, 'face_train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'face_test.csv'))


# In[6]:


i = random.choice(train.index)
img_name = train.ID[i]
img = imread(os.path.join(data_dir, 'Train', img_name))
print('Age:', train.Class[i])
imshow(img)


# In[7]:


from scipy.misc import imresize

temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (32,32))
    img = img.astype('float32')
    temp.append(img)
    
train_x = np.stack(temp)


# In[8]:


temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (32,32))
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)


# In[9]:


train.Class.value_counts(normalize=True)


# In[10]:


import keras


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)
train_y


# In[17]:


i1 = (32,32, 3)
h1 = 500
o1 = 3

e1 = 5
b1 = 120

from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

model = Sequential([
    InputLayer(input_shape = i1),
    Flatten(),
    Dense(units=h1, activation='relu'),
    Dense(units=o1, activation='softmax'),
])

model.summary()


# In[18]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_x, train_y, batch_size=b1, epochs=e1, verbose=1, validation_split=0.2)


# In[ ]:




