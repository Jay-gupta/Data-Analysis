#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[4]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[7]:


from nltk.corpus import stopwords


# In[8]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[9]:


#import gensim


# In[22]:


df  = pd.read_csv("C:/Users/hkc03/Desktop/JAY/tweets.csv")
df


# In[12]:


pd.set_option('display.max_colwidth', -1)


# In[50]:


df_short = df[['text', 'airline_sentiment']]


# In[37]:


df_short


# In[16]:


import re


# In[51]:


c = pd.DataFrame({'count_mentions': df_short.text.str.count('@\w+')})


# In[52]:


df_short = pd.concat([df_short, c], axis=1, join_axes=[df_short.index])


# In[53]:


df_short


# In[54]:


c = pd.DataFrame({'count_hashtags': df_short.text.str.count('#\w+')})


# In[55]:


df_short = pd.concat([df_short, c], axis=1, join_axes=[df_short.index])


# In[56]:


df_short


# In[57]:


c = pd.DataFrame({'count_words': df_short.text.str.count('\w+')})


# In[58]:


df_short = pd.concat([df_short, c], axis=1, join_axes=[df_short.index])


# In[59]:


df_short


# In[60]:


c = pd.DataFrame({'count_exclamations': df_short.text.str.count('!|\?+')})


# In[61]:


df_short = pd.concat([df_short, c], axis=1, join_axes=[df_short.index])


# In[62]:


df_short


# In[63]:


#Capital Words, Urls, emojis 'import emoji'


# In[68]:


c = pd.DataFrame({'count_capitals': df_short.text.str.count('[A-Z]{2,}')})


# In[70]:


df_short = pd.concat([df_short, c], axis=1, join_axes=[df_short.index])


# In[71]:


df_short


# In[74]:


c = pd.DataFrame({'count_capitals': df_short.text.str.count('^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$')})


# In[77]:


#filter
is_pos = df['airline_sentiment'].str.contains('positive')
positive_tweets = df[is_pos]
positive_tweets.shape


# In[78]:


is_pos = df['airline_sentiment'].str.contains('negative')
negative_tweets = df[is_pos]
negative_tweets.shape


# In[80]:


is_pos = df['airline_sentiment'].str.contains('neutral')
neutral_tweets = df[is_pos]
neutral_tweets.shape


# In[89]:


worst_airline = negative_tweets[['airline', 'airline_sentiment_confidence']]
worst_airline


# In[83]:


count_worst_airline = worst_airline.groupby('airline', as_index =False).count()
count_worst_airline.sort_values('airline_sentiment_confidence', ascending=False)


# In[88]:


best_airline = positive_tweets[['airline', 'airline_sentiment_confidence']]
count_best_airline = best_airline.groupby('airline', as_index =False).count()
count_best_airline.sort_values('airline_sentiment_confidence', ascending=False)


# In[91]:


reason = negative_tweets[['airline', 'negativereason']]
#reason['negativereason'].value_counts()
count_flight_reason = reason.groupby('negativereason', as_index=False).count()
count_flight_reason.sort_values('negativereason', ascending = True)


# In[92]:


#nltk
#naive-bayesian classifier

import nltk


# In[93]:


nltk.download('punkt')


# In[94]:


nltk.download('stopwords')


# In[95]:


import string


# In[96]:


string.punctuation


# In[98]:


useless_words = nltk.corpus.stopwords.words('english')
useless_words


# In[104]:


tokenized_negative_tweets = []
for text in negative_tweets['text']:
    tokenized_negative_tweets.append(nltk.word_tokenize(text))
    
def remove_useless_words(words):
    return{
        word: 1 for word in words if not word in useless_words
    }


# In[130]:


negative_clean = [(remove_useless_words(text), 'neg') for text in tokenized_negative_tweets]
negative_clean


# In[131]:


tokenized_positive_tweets = []
for text in positive_tweets['text']:
    tokenized_positive_tweets.append(nltk.word_tokenize(text))
    
def remove_useless_words(words):
    return{
        word: 1 for word in words if not word in useless_words
    }


# In[132]:


positive_clean = [(remove_useless_words(text), 'pos') for text in tokenized_positive_tweets]
positive_clean


# In[109]:


from nltk.classify import NaiveBayesClassifier


# In[133]:


len(negative_clean)


# In[134]:


len(positive_clean)


# In[123]:


split=2000


# In[135]:


sc = NaiveBayesClassifier.train(positive_clean[:split]+negative_clean[:split])


# In[136]:


nltk.classify.util.accuracy(sc, positive_clean[:split]+negative_clean[:split])


# In[137]:


positive_verify = positive_clean[split:]
negative_verify = negative_clean[split:]


# In[138]:


nltk.classify.util.accuracy(sc, positive_verify+negative_verify)


# In[139]:


is_customerservice = negative_tweets['negativereason'].str.contains('Customer Service Issue')
customer_service = negative_tweets[is_customerservice]


# In[140]:


tokenize = []
for text in customer_service['text']:
    tokenize.append(nltk.word_tokenize(text))
tokenize


# In[141]:


clean = [(remove_useless_words(text), 'pos') for text in tokenize]


# In[142]:


classifier = NaiveBayesClassifier.train(clean[:1000])


# In[143]:


nltk.classify.util.accuracy(classifier, clean[1000:])


# In[144]:


sc.show_most_informative_features()


# In[145]:


sc.labels()


# In[147]:


test_tweet = "I will never fly your planes again"
tokenize = []
for text in test_tweet:
    tokenize.append(nltk.word_tokenize(text))
tokenize

clean = [remove_useless_words(text) for  in tokenize]

sc.classify(clean)


# In[ ]:




