#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from scipy.io import arff

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


data = arff.loadarff('vote.arff')


# In[3]:


df = pd.DataFrame(data[0])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.sort_index(inplace=True)
encoder = LabelEncoder()
df.tail()


# In[7]:


df.sort_index(inplace=True)
vectorizer = CountVectorizer()
all_features = vectorizer.fit_transform(df.Class)
all_features.shape


# In[8]:


df['handicappedinfants'] = encoder.fit_transform(df.handicappedinfants.values)
df['waterprojectcostsharing'] = encoder.fit_transform(df.waterprojectcostsharing.values)
df['adoptionofthebudgetresolution'] = encoder.fit_transform(df.adoptionofthebudgetresolution.values)
df['physicianfeefreeze'] = encoder.fit_transform(df.physicianfeefreeze.values)
df['elsalvadoraid'] = encoder.fit_transform(df.elsalvadoraid.values)
df['religiousgroupsinschools'] = encoder.fit_transform(df.religiousgroupsinschools.values)
df['antisatellitetestban'] = encoder.fit_transform(df.antisatellitetestban.values)
df['aidtonicaraguancontras'] = encoder.fit_transform(df.aidtonicaraguancontras.values)
df['mxmissile'] = encoder.fit_transform(df.mxmissile.values)
df['immigration'] = encoder.fit_transform(df.immigration.values)
df['synfuelscorporationcutback'] = encoder.fit_transform(df.synfuelscorporationcutback.values)
df['educationspending'] = encoder.fit_transform(df.educationspending.values)
df['superfundrighttosue'] = encoder.fit_transform(df.superfundrighttosue.values)
df['crime'] = encoder.fit_transform(df.crime.values)
df['dutyfreeexports'] = encoder.fit_transform(df.dutyfreeexports.values)
df['exportadministrationactsouthafrica'] = encoder.fit_transform(df.exportadministrationactsouthafrica.values)


# In[9]:


df.tail()


# In[10]:


x = df.drop(['Class'], axis = 1)
y = df.handicappedinfants
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO1.txt','w',encoding="utf8")
f3.write("Atributo -> handicappedinfants\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO1.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[11]:


x = df.drop(['Class'], axis = 1)
y = df.waterprojectcostsharing
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO2.txt','w',encoding="utf8")
f3.write("Atributo -> waterprojectcostsharing\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO2.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO2.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[12]:


x = df.drop(['Class'], axis = 1)
y = df.adoptionofthebudgetresolution
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO3.txt','w',encoding="utf8")
f3.write("Atributo -> adoptionofthebudgetresolution\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO3.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO3.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[13]:


x = df.drop(['Class'], axis = 1)
y = df.physicianfeefreeze
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO4.txt','w',encoding="utf8")
f3.write("Atributo -> physicianfeefreeze\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO4.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO4.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[14]:


x = df.drop(['Class'], axis = 1)
y = df.elsalvadoraid
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO5.txt','w',encoding="utf8")
f3.write("Atributo -> elsalvadoraid\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO5.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO5.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[15]:


x = df.drop(['Class'], axis = 1)
y = df.religiousgroupsinschools
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO6.txt','w',encoding="utf8")
f3.write("Atributo -> religiousgroupsinschools\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO6.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO6.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[16]:


x = df.drop(['Class'], axis = 1)
y = df.antisatellitetestban
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO7.txt','w',encoding="utf8")
f3.write("Atributo -> antisatellitetestban\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO7.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO7.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[17]:


x = df.drop(['Class'], axis = 1)
y = df.aidtonicaraguancontras
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO8.txt','w',encoding="utf8")
f3.write("Atributo -> aidtonicaraguancontras\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO8.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO8.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[18]:


x = df.drop(['Class'], axis = 1)
y = df.mxmissile
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO9.txt','w',encoding="utf8")
f3.write("Atributo -> mxmissile\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO9.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO9.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[19]:


x = df.drop(['Class'], axis = 1)
y = df.immigration
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO10.txt','w',encoding="utf8")
f3.write("Atributo -> immigration\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO10.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO10.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[20]:


x = df.drop(['Class'], axis = 1)
y = df.synfuelscorporationcutback
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO11.txt','w',encoding="utf8")
f3.write("Atributo -> synfuelscorporationcutback\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO11.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO11.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[21]:


x = df.drop(['Class'], axis = 1)
y = df.educationspending
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO12.txt','w',encoding="utf8")
f3.write("Atributo -> educationspending\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO12.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO12.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[22]:


x = df.drop(['Class'], axis = 1)
y = df.superfundrighttosue
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO13.txt','w',encoding="utf8")
f3.write("Atributo -> superfundrighttosue\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO13.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO13.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[23]:


x = df.drop(['Class'], axis = 1)
y = df.crime
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO14.txt','w',encoding="utf8")
f3.write("Atributo -> crime\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO14.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO14.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[24]:


x = df.drop(['Class'], axis = 1)
y = df.dutyfreeexports
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO15.txt','w',encoding="utf8")
f3.write("Atributo -> dutyfreeexports\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO15.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO15.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()


# In[25]:


x = df.drop(['Class'], axis = 1)
y = df.exportadministrationactsouthafrica
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=18)

classifier = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
f3 = open ('cassReportSMO16.txt','w',encoding="utf8")
f3.write("Atributo -> exportadministrationactsouthafrica\n\n======================================\n"+"Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))
f3.close()
print("Porcentaje, {:.2f}".format(classifier.score(X_test, y_test)))

f3 = open ('cassReportSMO16.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Classification Report\n\n" +str(classification_report(y_test,y_pred)))
f3.close()

f3 = open ('cassReportSMO16.txt','a',encoding="utf8")
f3.write("\n======================================\n"+"Confusion Matrix\n\n"+str(confusion_matrix(y_test, y_pred)))
f3.close()

