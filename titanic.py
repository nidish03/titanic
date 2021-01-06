#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


data = pd.read_csv('D:\\nidishdoc\\titanic\\titanic.csv')
data = data.drop('PassengerId', axis = 1)
data


# In[4]:


missing=data.isnull().sum()
missing


# In[36]:


data_1=data.copy()
data_1['Age']=data_1['Age'].fillna(data_1['Age'].mean())
unique=data_1['Cabin'].unique()
unique
unique_1=data_1['Cabin'].value_counts()
unique_1


# In[37]:


data_1['Cabin']=data_1['Cabin'].fillna(data_1['Cabin'].mode()[0])
data_1['Cabin']
data_1.info()


# In[38]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data_1['Cabin']= label_encoder.fit_transform(data_1['Cabin'])
data_1.head()


# In[39]:


correlation=data_1.corr()
sns.heatmap(correlation,annot=True)
plt.show()


# In[40]:


sns.pairplot(data_1)
plt.show()


# In[71]:


columns=['Embarked','Sex']
def categories(multi_columns):
    final=data_1
    i=0
    for field in multi_columns:
        
        print(field)
        data_2=pd.get_dummies(data_1[field],drop_first=True)
        data_1.drop([field],axis=1,inplace=True)
        if i == 0:
            final=data_2.copy()
        else:
            final=pd.concat([final,data_2],axis=1)
        i=i+1
    final=pd.concat([data_1,final],axis=1)
    
    return final


# In[72]:


final_data_1=categories(columns)


# In[73]:


final_data_1


# In[77]:


final_data=final_data_1.drop(columns=["Name",'Ticket'],axis=1)
final_data['Age']=final_data.Age.astype(int)
final_data.info()


# In[86]:


X=final_data.drop(['Survived'],axis=1)
Y=final_data['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
X_train


# In[87]:


logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred= logistic.predict(X_test)
score_1=accuracy_score(y_test,y_pred)
print("Accuracy on Traing set: ",logistic.score(X_train,y_train))
print("Accuracy on Testing set: ",logistic.score(X_test,y_test))
print("accuracy_score", score_1)


# In[ ]:




