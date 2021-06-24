#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os


# In[2]:


from tensorflow.keras import layers,models


# ## First woking with the train data
# 

# In[3]:


import pandas  as pd
train_data = pd.read_csv('train.csv')


# In[4]:


train_data.head()


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data['Condition'].value_counts()


# In[7]:


from imblearn.under_sampling import NearMiss


# In[8]:


train_data_2 = train_data.copy()


# In[9]:


train_data_2['index'] = range(0,len(train_data_2))


# In[10]:


train_data_2


# In[11]:


train_data_2.columns


# In[12]:


train_data['Min_coverage'].fillna(train_data['Min_coverage'].median(), inplace=True)


# In[13]:


train_data['Cost_of_vehicle'].fillna(train_data['Cost_of_vehicle'].median(), inplace=True)


# In[14]:


train_data['Max_coverage'].fillna(train_data['Max_coverage'].median(), inplace=True)


# In[15]:


train_data.describe()


# In[16]:


train_data['Amount'].fillna(train_data['Amount'].median(), inplace=True)


# In[17]:


train_data.isnull().sum()


# In[18]:


train_data_2['Cost_of_vehicle'].fillna(train_data_2['Cost_of_vehicle'].median(), inplace=True)


# In[19]:


train_data_2['Max_coverage'].fillna(train_data_2['Max_coverage'].median(), inplace=True)


# In[20]:


train_data_2['Amount'].fillna(train_data_2['Amount'].median(), inplace=True)


# In[21]:


train_data_2['Min_coverage'].fillna(train_data_2['Min_coverage'].median(), inplace=True)


# In[22]:


train_data_2.isnull().sum()


# In[23]:


X = train_data_2.drop(['Image_path', 'Insurance_company','Condition','Expiry_date'],axis=1).values


# In[24]:


y = train_data_2['Condition'].values


# In[25]:


X.shape


# In[26]:


y.shape


# In[27]:


X


# In[28]:


nm = NearMiss()
X_res,y_res=nm.fit_sample(X,y)


# In[29]:


X_res.shape


# In[30]:


y_res.shape


# In[31]:


dataframe = pd.DataFrame(X_res)  


# In[32]:


dataframe['Condition'] = y_res


# In[33]:


dataframe


# In[34]:


train_data_2.columns


# In[35]:


dataframe.columns=['Cost_of_vehicle', 'Min_coverage', 'Max_coverage', 'Amount', 'index', 'Condition' ]


# In[36]:


dataframe


# In[37]:


conditions=[]


# In[38]:


for i in range(len(dataframe)):
    row=dataframe.iloc[i]
    path=row['index']
    dft=train_data_2.loc[train_data_2['index']==path]
    condition=dft.values[0]
    conditions.append(condition)


# In[39]:


conditions


# In[40]:


df = pd.DataFrame(conditions)


# In[41]:


df


# In[42]:


df.columns = ['A','b','c','d','e','f','g','h','i']


# In[43]:


df


# In[44]:


df_original = pd.DataFrame()


# In[45]:


df_original['Image_path'] = df['A']


# In[46]:


df_original['Conditions'] = df['g']   


# In[47]:


df_original


# In[48]:


y_train = df_original['Conditions'].values


# In[49]:


y_train=  y_train.reshape(-1,1)


# In[50]:


y_train


# ## Now getting the images 

# In[51]:


data_dir = r"C:\Users\The ChainSmokers\dataset"


# In[52]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[53]:


train = list(data_dir.glob('trainImages/*.jpg'))
train[:5]


# In[54]:


lol=[]


# In[55]:


for i in df_original['Image_path']:
    lol.append(i)


# In[56]:


lol


# In[57]:


first = r"C:/Users/The ChainSmokers/dataset/trainImages/"


# In[58]:


ne_lol = [first + x for x in lol]


# In[59]:


ne_lol


# In[60]:


fir = ne_lol[5]


# In[61]:


Image.open(fir)


# In[ ]:


X_train=[]


# In[ ]:


for imgage in ne_lol:
    img = cv2.imread(str(imgage))
    resized_img = cv2.resize(img, (180,180))
    X_train.append(resized_img)
    


# In[ ]:


X_train


# In[ ]:


X_train = np.array(X_train)


# In[ ]:


X_train = X_train/255


# In[ ]:


X_train.shape


# ## CNN MODEL 

# In[ ]:


cnn = models.Sequential([
    layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu', input_shape=(180, 180, 3)),
    layers.MaxPooling2D((1, 1)),
    
    layers.Conv2D(filters=30, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((1, 1)),
    
    layers.Flatten(),
    layers.Dense(40, activation='relu'),
    layers.Dense(2, activation='softmax')
])


# In[ ]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


cnn.fit(X_train, y_train, epochs=10)


# In[ ]:


cnn.save('CNN_Model.model')


# ## Now working with the prediction of Amount

# In[ ]:


df


# In[ ]:


df.drop(['A','i'],axis=1, inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.drop(['e'],axis=1,inplace=True)


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


df


# In[ ]:


X_ann =df.drop(['h'], axis=1).values


# In[ ]:


y_ann  = df['h'].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_ann = sc.fit_transform(X_ann)


# In[ ]:


X_ann


# ## Building an Xgboost model
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)


# In[ ]:


regressor.fit(X_ann, y_ann)


# In[ ]:


dd = pd.read_csv('dekh_le.csv')


# In[ ]:


dd


# In[ ]:


dd.columns


# In[ ]:


dd.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[ ]:


X_test_ann = dd.values


# In[ ]:


X_test_ann


# In[ ]:


predict_lol = regressor.predict(X_test_ann)


# In[ ]:


predict_lol = yesabhai.reshape(-1,1)


# In[ ]:


predict_lol


# In[ ]:


tt = pd.read_csv('test.csv')


# In[ ]:


submission = pd.DataFrame(tt['Image_path'])


# In[ ]:


submission


# In[ ]:


ff = pd.read_csv('ff.csv')


# In[ ]:


ff


# In[ ]:


submission['Condition'] = ff['Condition']


# In[ ]:


submission


# In[ ]:


submission['Amount'] = predict_lol


# In[ ]:


submission


# In[ ]:


submission.to_csv('final.csv')


# In[ ]:




