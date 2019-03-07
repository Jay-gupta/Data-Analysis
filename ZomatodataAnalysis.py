#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[1]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[5]:


data = pd.read_csv('C:/Users/hkc03/Desktop/JAY/project2/zomato.csv')
data


# In[12]:


data.describe()


# In[9]:


data['Country Code'].value_counts()


# In[10]:


data['Cuisines'].value_counts()


# In[15]:


country = data[data['Country Code'] == 1]
country


# In[17]:


color_rating = country[country['Rating color'] =="Dark Green"]
color_rating


# In[24]:


result = data[(data['Country Code'] == 1) & (data['Rating color'] =="Dark Green") & (data['Aggregate rating']>4.5)]
result


# In[20]:


x = "hello"


# In[21]:


del x


# In[22]:


x


# In[25]:


result['Has Table booking'].value_counts()


# In[26]:


result['Has Online delivery'].value_counts()


# In[27]:


result['Is delivering now'].value_counts()


# In[ ]:




