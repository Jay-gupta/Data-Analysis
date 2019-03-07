#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import pandas as pd


# In[ ]:





# In[2]:


a = [1,2,3,4,5]
a


# In[3]:


b = a[2:4]
b


# In[4]:


b[0] = 7
b


# In[5]:


a


# In[7]:


a = np.array(a)
a


# In[8]:


b = a[2:4]
b


# In[9]:


b[0] = 7
b


# In[10]:


a


# creating copies of array in numpy

# In[11]:


e = a[2:4].copy()
e


# In[13]:


e[0]=8
e


# In[14]:


a


# In[16]:


x = [1,2,3,4,5,6,7,8,9]
x = np.array(x)
x1 = x.reshape((3,3))
x1


# In[20]:


x = [1,2,3]
y = [4,5,6]
z = [7,8,9]

c = np.vstack([[np.concatenate([x,y])],[np.concatenate([y,z])]])
c


# In[25]:


np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 100, size=1000000)
get_ipython().run_line_magic('timeit', 'compute_reciprocals(values)')


# In[33]:


x = np.arange(1, 6)
print(x)
np.add.reduce(x)


# In[34]:


m = np.random.random((3,4))
m


# In[36]:


m.sum(axis=1)


# In[5]:


t = np.random.random((10))
t


# In[6]:


t = t*10
t


# In[7]:


x = np.arange(1,11)
x


# In[8]:


plt.scatter(x, t, alpha=0.5)
plt.show()


# In[11]:


data = pd.DataFrame(t,index=x)
data


# In[10]:


test = pd.read_csv('C:/Users/hkc03/Desktop/JAY/project2/test_Y3wMUE5_7gLdaTN.csv')
test


# In[12]:


test.describe()


# In[13]:


train = pd.read_csv('C:/Users/hkc03/Desktop/JAY/project2/train_u6lujuX_CVtuZ9i.csv')
train


# In[14]:


train['Property_Area'].value_counts()


# In[15]:


train.boxplot(column='ApplicantIncome', by = 'Education')


# In[18]:


temp1 = train['Credit_History'].value_counts(ascending=True)


# In[19]:


temp2 = train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())


# In[20]:


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')


# In[21]:


ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[22]:


temp3 = pd.crosstab(train['Credit_History'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[ ]:




