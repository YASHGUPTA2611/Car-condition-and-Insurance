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


test_data=pd.read_csv('test.csv')


# In[3]:


test_data.head()


# In[4]:


test_data.isnull().sum()


# In[5]:


data_dir = r"C:\Users\The ChainSmokers\dataset"


# In[6]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[7]:


test = list(data_dir.glob('testImages\*.jpg'))


# In[8]:


test


# In[9]:


lol_test =[]


# In[10]:


for i in test_data['Image_path']:
    lol_test.append(i)


# In[11]:


lol_test


# In[12]:


first = r"C:/Users/The ChainSmokers/dataset/testImages/"


# In[13]:


full_path = [first + x for x in lol_test]


# In[14]:


full_path


# In[15]:


X_test=[]


# In[16]:


for imgage in full_path:
    img = cv2.imread(str(imgage))
    resized_img = cv2.resize(img, (180,180))
    X_test.append(resized_img)
    


# In[ ]:


X_test


# In[ ]:


X_test = np.array(X_test)


# In[ ]:


X_test = X_test/255


# In[ ]:


X_test.shape


# In[ ]:


new_model = tf.keras.models.load_model(r'C:\Users\The ChainSmokers\dataset\CNN_Model.model')


# In[ ]:


predictions = new_model.predict(X_test)


# In[ ]:


predictions


# In[ ]:


y_prediction_labels = [np.argmax(i) for i in predictions]


# In[ ]:


y_prediction_labels


# ## Test data prediction

# In[ ]:


test_data_copy = test_data.copy()


# In[ ]:


test_data_copy


# In[ ]:


test_data_copy['Condition'] =  y_prediction_labels


# In[ ]:


test_data_copy


# In[ ]:


test_data_copy.columns


# In[ ]:


test_data_copy.drop(['Image_path','Expiry_date'], axis=1, inplace=True)


# In[ ]:


test_data_copy


# In[ ]:


test_data_copy = pd.get_dummies(test_data_copy)


# In[ ]:


X_test_ann = test_data_copy.values


# In[ ]:


X_test_ann


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test_ann = sc.fit_transform(X_test_ann)


# In[ ]:


X_test_ann


# In[ ]:


dd = pd.DataFrame(X_test_ann)


# In[ ]:


dd


# In[ ]:


dd.to_csv('dekh_le.csv')


# In[ ]:


test_data_copy['Condition'].to_csv('ff.csv')


# In[ ]:




