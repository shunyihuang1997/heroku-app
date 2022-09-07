#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import requests
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import mean_squared_error
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence



# # EDA

# In[2]:


def eda(df):
    print(df.head(2),'\n')
    print(df.info())
    print(df.describe(),'\n')
    print(df.isna().sum())
    print('Number of data point is: ', len(df))

def box_plot(df, feature):
   df.plot(y = feature, kind = 'box')

def heat_map(df):
    sns.heatmap(df.corr(),annot=True)
    
def check_vif(df,features):
    df = df.drop(features,axis = 1)
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    print(vif_data)


# In[3]:


df = pd.DataFrame(pd.read_csv('ds_salaries.csv'))
#df.head()


# In[4]:


le = LabelEncoder()
df['salary_currency'] = le.fit_transform(df['salary_currency'])
df['employee_residence'] = le.fit_transform(df['employee_residence'])
df['company_location'] = le.fit_transform(df['company_location'])
df['company_size'] = le.fit_transform(df['company_size'])
df['experience_level'] = le.fit_transform(df['experience_level'])
df['job_title'] = le.fit_transform(df['job_title'])


# In[5]:


df['employment_type'] = le.fit_transform(df['employment_type'])
df.head()


# In[6]:


# eda(df)
# heat_map(df)
# check_vif(df,['work_year','company_location','salary_currency'])


# # Data splitting

# In[7]:
X = df.loc[:, df.columns != 'salary_in_usd']
X = X.drop(['Unnamed: 0','work_year','salary','salary_currency','company_location'],axis = 1)
y = df['salary_in_usd']


# df = df.drop(['work_year','company_location','salary_currency','salary'],axis = 1)
# X = df.loc[:, df.columns != 'salary_in_usd']
# X = X.drop(['Unnamed: 0','experience_level','employment_type','employee_residence','remote_ratio','company_size'],axis = 1)
# y = df['salary_in_usd']
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)


# In[8]:


X.head()


# # Modeling

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)
lm = LinearRegression()
lm.fit(X_train, y_train)



# Saving model to disk
pickle.dump(lm, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))



