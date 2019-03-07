#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[1]:


import matplotlib
import matplotlib.pyplot as plt


# In[12]:


import numpy as np
np.__version__


# In[9]:


x = np.linspace(0, 10, 100)

fig = plt.figure()

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()


# In[4]:


print(x)


# In[10]:


fig.savefig('C:/Users/hkc03/Desktop/JAY/project2/my_figure.png')


# In[13]:


get_ipython().run_line_magic('pinfo', 'np')


# In[15]:


import pandas as pd


# In[16]:


pd.__version__


# In[27]:


x = [1, 2, 3, 4, 5]
x = np.array(x, float)
print(x)


# In[41]:


x = np.array(x, dtype='int32')
print(x)


# In[35]:


y = np.zeros((3,3,3), dtype='int')
print(y)


# In[38]:


np.eye(3, dtype='int')


# In[42]:


np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array


# In[43]:


x1.dtype


# In[ ]:




