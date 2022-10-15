#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy.fft as nf
from scipy import stats
from sklearn.linear_model import LinearRegression


# In[2]:


def filter_data(df):
    b, a = signal.butter(3, 0.15, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, df['aT (m/s^2)'])
    #peaks,_ = signal.find_peaks(df['aT (m/s^2)'])
    #plt.plot(df['time'],low_passed)
    #plt.plot(df['time'],df['aT (m/s^2)'])
    return low_passed


def calculate_vel(df):
    temp=df.shift(periods=-1)
    diff_df=abs(df-temp)
    temp_df=diff_df.shift(periods=1)
    temp_df.iloc[0]=df.iloc[0]
    df['diff_time']=temp_df['time']
    df['velocity']=df['diff_time']*df['aT (m/s^2)']
    return df
    

def calculate_delta_position(df):
    df['position']=df['diff_time']*df['velocity']
    return df

def calculate_total_position(df):
    df2=df
    df2=df2.cumsum(axis=0)
    df['total_position']=df2['position']
    return df


def transform_df(df):
    #using the above function to form the final df we want
    df['aT (m/s^2)']=filter_data(df)
    df=calculate_vel(df)
    df=calculate_delta_position(df)
    df=calculate_total_position(df)
    return df

    

def get_model(df):
    X=np.stack([df['time']],axis=1)
    X_with=np.concatenate([np.ones(X.shape),X],axis=1)
    model=LinearRegression(fit_intercept=False)
    model.fit(X_with,df['total_position'])
    return model
    


# In[3]:


#foot model:10 mins
df=pd.read_csv('foot') 
df=transform_df(df)
model=get_model(df)
#print(model.coef_)


# In[4]:


#enter the right foot
right=pd.read_csv('right_foot')
right=transform_df(right)
X=np.stack([right['time']],axis=1)
X_with=np.concatenate([np.ones(X.shape),X],axis=1)
print(model.score(X_with,right['total_position']))


# In[5]:


#enter the injury csv 
injury=pd.read_csv('injury')
injury=transform_df(injury)
X2=np.stack([injury['time']],axis=1)
X2_with=np.concatenate([np.ones(X2.shape),X2],axis=1)
print(model.score(X2_with,injury['total_position']))


# In[6]:


#enter the left foot
left=pd.read_csv('left_foot')
left=transform_df(left)
X3=np.stack([left['time']],axis=1)
X3_with=np.concatenate([np.ones(X3.shape),X3],axis=1)
print(model.score(X3_with,left['total_position']))


# In[7]:


#right foot model: 3mins
model_right=get_model(right)
print(model_right.score(X3_with,left['total_position']))#left foot score
print(model_right.score(X2_with,injury['total_position']))#injury score


# In[ ]:




