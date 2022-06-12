#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np 
from scipy.io import arff

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import csv


# In[21]:


data = pd.read_csv("train.csv")


# In[22]:


df = pd.DataFrame(data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[23]:


df.shape


# In[24]:


df.sort_index(inplace=True)
encoder = LabelEncoder()
df.tail()


# In[25]:


set(data.ClassEmociones)


# In[26]:


df.sort_index(inplace=True)
vectorizer = CountVectorizer()
all_features = vectorizer.fit_transform(df.ClassEmociones)
all_features.shape


# In[27]:


df['textosEads'] = encoder.fit_transform(df.textosEads.values)
df['ClassEmociones'] = encoder.fit_transform(df.ClassEmociones.values)


# In[28]:


df.tail()


# In[46]:


x = df.drop(['textosEads'], axis = 1)
y = df.ClassEmociones
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.5, random_state=50)
X_train.shape
X_test.shape

classifier = RandomForestClassifier(max_depth=6, random_state=50)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportRF1.txt','w',encoding="utf8")
f3.write("Atributo -> textosEads\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

#Immprimir tabla en el txt
f3 = open ('cassReportRF1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportRF1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(set(data.ClassEmociones))+"\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[ ]:




