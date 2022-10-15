#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy.fft as nf
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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
    df = df[0:16000]

    return df


def get_model(df):
    X=np.stack([df['time']],axis=1)
    X_with=np.concatenate([np.ones(X.shape),X],axis=1)
    model=LinearRegression(fit_intercept=False)
    model.fit(X_with,df['total_position'])
    return model
    

def get_model_train_set(df):
    X=np.stack([df['time']],axis=1)
    X_with=np.concatenate([np.ones(X.shape),X],axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_with,df['total_position'])
    return X_train, X_valid, y_train, y_valid


# In[3]:


#foot model:10 mins
df=pd.read_csv('foot') 
df=transform_df(df)
model_long=get_model(df)

#print(model.coef_)

# foot model:3 mins
right=pd.read_csv('right_foot')
right=transform_df(right)
model_short=get_model(right)


# In[4]:


list_of_title = np.array(['right_foot_1','right_foot_2','right_foot_3','female_1','female_2',
                                    'injury','injury_1','injury_2'])
list_of_situation = np.array(['0','0','0','0','0','1','1','1',])

list_of_whole = []
for i in range(8):
    input_file = pd.read_csv(list_of_title[i])
    input_file = transform_df(input_file)
    
    X_train, X_valid, y_train, y_valid = get_model_train_set(input_file)
    
    print("The injury situation of input",list_of_title[i], "is ", list_of_situation[i])
    temp1 = model_short.score(X_train,y_train)
    temp2 = model_short.score(X_valid,y_valid)
    print("The train socre of input",list_of_title[i], "is ", temp1)
    print("The valid socre of input",list_of_title[i], "is ", temp2)
    list_of_whole.append([])
    list_of_whole[i].append(temp1)
    list_of_whole[i].append(temp2)
    list_of_whole[i].append(list_of_situation[i])
    


# In[5]:


# In[6]:


dataframe = pd.DataFrame(list_of_whole,columns = ['train_score','valid_score','situation'],index = ['right_foot_1','right_foot_2','right_foot_3',
                                                        'female_1','female_2','injury','injury_1','injury_2'])

# In[7]:


#plug the dataframe into classifier
G_model=GaussianNB()
X=dataframe[['train_score','valid_score']]
y=dataframe['situation']
G_model.fit(X,y)
# print(G_model.theta_)
# print(G_model.sigma_)


# In[9]:


def main():
    #test the model working state， and worked
    test = pd.read_csv(sys.argv[1])
    test=transform_df(test)
    X_train, X_valid, y_train, y_valid = get_model_train_set(test)
    temp1 = model_short.score(X_train,y_train)
    temp2 = model_short.score(X_valid,y_valid)
    X_check=[[temp1,temp2]]
    G_model.predict(X_check)
    print("\n")
    print("The result of input file in normal classifier is (Note 1 means injury; 0 means not injury) ", G_model.predict(X_check))
    

if __name__ == '__main__':
    main()




