#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np 
from scipy.io import arff

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import csv


# In[8]:


df = pd.read_csv("train.csv")


# In[9]:


#df = pd.DataFrame(data[0])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[10]:


df.tail()


# In[11]:


df.shape


# In[12]:


df.sort_index(inplace=True)
encoder = LabelEncoder()
df.tail()


# In[14]:


df.sort_index(inplace=True)
vectorizer = CountVectorizer()
all_features = vectorizer.fit_transform(df.ClassEmociones)
all_features.shape


# In[15]:


df['textosEads'] = encoder.fit_transform(df.textosEads.values)
df['ClassEmociones'] = encoder.fit_transform(df.ClassEmociones.values)


# In[25]:


set(data.ClassEmociones)


# In[16]:


df.tail()


# In[28]:


x = df.drop(['textosEads'], axis = 1)
y = df.ClassEmociones
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.5, random_state=30)
X_train.shape
X_test.shape

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportNB1.txt','w',encoding="utf8")
f3.write("Atributo -> Textos\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

#Immprimir tabla en el txt
f3 = open ('cassReportNB1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportNB1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(set(data.ClassEmociones))+"\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[ ]:




