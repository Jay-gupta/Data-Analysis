#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pandas as pd


# In[4]:


import datetime


# In[5]:


test = pd.read_csv('C:/Users/hkc03/Desktop/JAY/project2/test_Y3wMUE5_7gLdaTN.csv')
test


# In[6]:


test.describe()


# In[7]:


train = pd.read_csv('C:/Users/hkc03/Desktop/JAY/project2/train_u6lujuX_CVtuZ9i.csv')
train


# In[8]:


train['Property_Area'].value_counts()


# In[13]:


train['Property_Area'].hist(bins=50)


# In[15]:


print("hello")


# In[8]:


dti = pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01'), datetime.datetime(2018, 1, 1)])


# In[9]:


dti


# In[10]:


dti = pd.date_range('2018-01-01', periods=3, freq='H')


# In[11]:


dti


# In[12]:


dti = dti.tz_localize('UTC')


# In[14]:


dti


# In[15]:


dti.tz_convert('US/Pacific')


# In[16]:


idx = pd.date_range('2018-01-01', periods=5, freq='H')


# In[17]:


idx


# In[18]:


ts = pd.Series(range(len(idx)), index=idx)


# In[19]:


ts


# In[20]:


ts.resample('2H').mean()


# In[21]:


x = pd.Timestamp('2018-01-05')


# In[22]:


x.day_name()


# In[23]:


y = x + pd.Timedelta('1 day')
x


# In[24]:


y


# In[25]:


pd.Series(range(3), index=pd.date_range('2000', freq='D', periods=3))


# In[30]:


pd.Series(pd.date_range('2000', freq='M', periods=10))


# In[27]:


pd.Series(pd.period_range('1/1/2011', freq='M', periods=3))


# In[31]:


pd.Series([pd.DateOffset(1), pd.DateOffset(2)])


# In[32]:


pd.Series(pd.date_range('1/1/2011', freq='M', periods=3))


# In[36]:


x1 = pd.Timestamp('2012-05-01')
x1


# In[37]:


x2 = pd.Timestamp('2012-05-20')
x2


# In[35]:


x2-x1


# In[38]:


np.random.seed(24)
df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
               axis=1)
df.iloc[0, 2] = np.nan


# In[39]:


df.style


# In[40]:


df.style.highlight_null().render().split('\n')[:10]


# In[41]:


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


# In[48]:


s = df.style.applymap(color_negative_red)
s


# In[43]:


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# In[46]:


df.style.apply(highlight_max)


# In[49]:


df.style.    applymap(color_negative_red).    apply(highlight_max)


# In[60]:


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


# In[67]:


df.style.apply(highlight_max, color='darkorange', axis=None)


# In[68]:


df.style.apply(highlight_max, color='darkorange', axis=0)


# In[69]:


df.style.apply(highlight_max, color='darkorange', axis=1)


# In[70]:


df.style.apply(highlight_max, subset=['B', 'C', 'D'])


# In[71]:


df.style.applymap(color_negative_red,
                  subset=pd.IndexSlice[2:5, ['B', 'D']])


# In[74]:


df.style.format("{:.2%}")


# In[75]:


df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})


# In[76]:


df.style.highlight_null(null_color='red')


# In[82]:


import seaborn as sns

cm = sns.light_palette("blue", as_cmap=True)

s = df.style.background_gradient(cmap=cm)
s


# In[84]:


df.style.    applymap(color_negative_red).    apply(highlight_max).    to_excel('styled.xlsx', engine='openpyxl')


#   

# PROGRAM TO HANDLE DATASET USING PANDAS

# In[61]:


test = pd.read_csv("C:/Users/hkc03/Desktop/JAY/project2/test_2AFBew7.csv")


# In[9]:


train = pd.read_csv("C:/Users/hkc03/Desktop/JAY/project2/train_gbW7HTd.csv")


# In[10]:


#univariate analysis

train.dtypes


# In[13]:


test.dtypes


# In[11]:


train


# In[12]:


test


# In[16]:


train.describe()


# In[17]:


categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index
print(categorical_variables)


# In[19]:


train[categorical_variables].apply(lambda x:len(x.unique()))


# In[21]:


train['Race'].value_counts()


# In[22]:


train['Race'].value_counts()/train.shape[0]


# In[44]:


train.shape


# In[23]:


train['Native.Country'].value_counts()


# In[24]:


train['Native.Country'].value_counts()/train.shape[0]


# #multivariate analysis
# 
# #continuous-continuous
# #categorical-categorical
# #continuous-categorical
# 
# 

#  

# In[34]:


#categorical-categorical , cross tabulation
ct = pd.crosstab(train['Sex'], train['Income.Group'], margins=True)
print(ct)


# In[29]:


ct.iloc[:-1, :-1].plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[33]:


def convertToPercent(series):
    return series/float(series[-1])

ct2 = ct.apply(convertToPercent, axis=1)
ct2.iloc[:-1, :-1].plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# #continuous-continuous

# In[36]:


train.plot('Age','Hours.Per.Week', kind = 'scatter')


# continuous-categorical

# In[37]:


#box blot

train.boxplot(column='Hours.Per.Week', by = 'Sex')


# MISSING VALUES
# 
# *Deletion
#     *list wise - delete entire row
#     *pair wise
# --> missing data is completely at random
# 
# *Mean/Median/Mode imputation
# *mean/median - generalised
# *mean/median - similar case imputation
# 
# *prediction model

# In[38]:


train.apply(lambda x:sum(x.isnull()))


# In[39]:


test.apply(lambda x:sum(x.isnull()))


# In[42]:


#mode values
#imputation

from scipy.stats import mode

mode(train['Workclass'].astype(str)).mode[0]


# In[47]:


cols = ['Workclass', 'Occupation', 'Native.Country']
for c in cols:
    train[c].fillna(mode(train[c].astype(str)).mode[0], inplace=True)
    test[c].fillna(mode(test[c].astype(str)).mode[0], inplace=True)


# In[48]:


train.apply(lambda x: sum(x.isnull()))


# 
# OUTLIER TREATMENT
# 
# -->univariate
# -->multivariate
# -->data entry
# -->measurement errors
# 
# *-1.5 IQR 1.5 IQR
# *How to remove?
# *delete observation
# 
# 

# In[49]:


#NUMERICAL
#SCATTER PLOT

train.plot('ID', 'Age', kind = 'scatter')


# 
# VARIABLE TRANSFORMATION

# In[50]:


train['Workclass'].value_counts()/train.shape[0]


# In[51]:


combine = ['State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked' ]
for cat in combine:
    train['Workclass'].replace({cat:'Others'}, inplace=True)
    test['Workclass'].replace({cat:'Others'}, inplace=True)


# In[52]:


train['Workclass'].value_counts()/train.shape[0]


# In[53]:


categorical = list(train.dtypes.loc[train.dtypes=='object'].index)
categorical


# In[54]:


categorical= categorical[1:]
categorical


# In[55]:


train[categorical].apply(lambda x:len(x.unique()))


# In[57]:


for column in categorical:
    f =train[column].value_counts()/train.shape[0]
    f2 = f.loc[f.values<0.05].index
    
    for c in f2:
        train[column].replace({c:'Others'}, inplace=True)
        test[column].replace({c:'Others'}, inplace=True)


# In[58]:


train[categorical].apply(lambda x:len(x.unique()))


# In[68]:


#Predictive modelling
#classification problem
#logistic regression, decision trees...

from sklearn.preprocessing import LabelEncoder
c = train.dtypes.loc[train.dtypes=='object'].index
c


# In[73]:


c = c[:-1]
c


# In[75]:


le  = LabelEncoder()
for a in c:
    #train[a] = le.fit_transform(train[a].astype(str))
    test[a] = le.fit_transform(test[a].astype(str))
    


# In[84]:


test


# In[71]:


train


# In[72]:


train.dtypes


# In[77]:


dependent_v = 'Income.Group'
independent_v = [x for x in train.columns if x not in [dependent_v, 'ID']]
independent_v


# In[76]:


from sklearn import tree


# In[79]:


model = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, max_features='sqrt')
model.fit(train[independent_v], train[dependent_v])


# In[80]:


p_train = model.predict(train[independent_v])


# In[81]:


from sklearn.metrics import accuracy_score


# In[83]:


acc_train = accuracy_score(train[dependent_v], p_train)
acc_train


# In[ ]:




