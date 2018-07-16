
# coding: utf-8

# In[243]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[244]:


data = pd.read_csv('data/train.csv')
a = len(data)
data.head()


# In[245]:


test_data = pd.read_csv('data/test.csv')
print(len(test_data))
test_data.head()


# In[246]:


target_label = data['Survived']
data = data.drop(['Survived'], axis = 1)
data.head()


# In[247]:


merged_data = pd.concat([data, test_data])


# In[248]:


print(len(merged_data))
merged_data.head()


# In[249]:


train_data = merged_data.drop(['Name','Ticket', 'Cabin'],axis = 1)


# In[250]:


train_data.head()


# In[251]:


cat_data = train_data.select_dtypes(include=['object'])


# In[252]:


cat_data.head()


# In[253]:


num_data = train_data.drop(cat_data, axis = 1)


# In[254]:


num_data.head()


# In[255]:


dummy_data = pd.get_dummies(cat_data)


# In[256]:


dummy_data.head()


# In[257]:


X = pd.concat([num_data, dummy_data], axis = 1)


# In[258]:


print(len(X))
X.head()


# In[259]:


X.isnull().sum()


# In[260]:


X = X.fillna(X.mean())
X.isnull().sum()


# In[261]:


x = X[:a]
y = X[a:]


# In[262]:


x.to_csv('Cleaned train data.csv')


# In[263]:


y.to_csv('Cleaned test data.csv')


# In[264]:


x_train, x_test, y_train, y_test = train_test_split(x, target_label, test_size = 0.10)


# In[265]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[266]:


scores = model.score(x_train, y_train)
scores


# In[267]:


y_pred = model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print('Model Accuracy {}'.format(round(accuracy * 100)))

test_data_prediction = model.predict(y)


# In[268]:


submission = pd.DataFrame({
    'PassengerId' : y['PassengerId'],
    'Survived' : test_data_prediction,
})

submission.to_csv('submission.csv', index = False)

