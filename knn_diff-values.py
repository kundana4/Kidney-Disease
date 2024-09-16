import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("kidney_disease.csv")
df.info()
print(df.isnull().sum())

# Data Pre-Processing
# '?' character remove process in the dataset
for i in ['rc','wc','pcv']:
    df[i] = df[i].str.extract('(\d+)').astype(float)

# Filling missing numeric data in the dataset with mean
for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    df[i].fillna(df[i].mean(),inplace=True)

df.info()
# Removing tab spaces in the data
df['dm'] = df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'}) 
df['cad'] = df['cad'].replace(to_replace='\tno',value='no') 
df['classification'] = df['classification'].replace(to_replace='ckd\t',value='ckd')

# Mapping the text to 1/0 and cleaning the dataset 
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0})
df['classification'] = df['classification'].replace(to_replace={'ckd':1,'notckd':0})
df.rename(columns={'classification':'class'},inplace=True)

df.drop('id',axis=1,inplace=True)
df.info()

# Filling the missing string data as the most repetitive (mod)
df=df.apply(lambda x:x.fillna(x.value_counts().index[0]))
df.info()

# Preparation of Model Data and Scaling of Data
features = [['age', 'bp','sg','al','su','bgr','bu', 'sc', 'sod','pot','hemo','pcv','wc', 'rc']]

# Scaling of the data
for feature in features:
    df[feature]=(df[feature]-np.min(df[feature]))/(np.max(df[feature])-np.min(df[feature]))

x_data=df.drop(['class'],axis=1)
y=df['class'].values

# Modelling
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x_data,y,test_size=0.3,random_state=42)

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# KNN with 3
knn=KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train,y_train)

knn_pred=knn.predict(x_test)
f1_knn=f1_score(y_test,knn_pred)
precision_knn = precision_score(y_test, knn_pred)
recall_knn = recall_score(y_test, knn_pred)

print("__________________________")
print("KNN 3")
print("____________________________")
print("KNN f1 score: ",f1_knn)
print("KNN Precision: ",precision_knn)
print("KNN Recall: ", recall_knn)
print("KNN accuracy score: ",knn.score(x_test,y_test))
print("____________________________")

# KNN with 6
knn=KNeighborsClassifier(n_neighbors=6) 
knn.fit(x_train,y_train)

knn_pred=knn.predict(x_test)
f1_knn=f1_score(y_test,knn_pred)
precision_knn = precision_score(y_test, knn_pred)
recall_knn = recall_score(y_test, knn_pred)

print("__________________________")
print("KNN 6")
print("____________________________")
print("KNN f1 score: ",f1_knn)
print("KNN Precision: ",precision_knn)
print("KNN Recall: ", recall_knn)
print("KNN accuracy score: ",knn.score(x_test,y_test))
print("____________________________")

# KNN with 9
knn=KNeighborsClassifier(n_neighbors=9) 
knn.fit(x_train,y_train)

knn_pred=knn.predict(x_test)
f1_knn=f1_score(y_test,knn_pred)
precision_knn = precision_score(y_test, knn_pred)
recall_knn = recall_score(y_test, knn_pred)

print("__________________________")
print("KNN 9")
print("____________________________")
print("KNN f1 score: ",f1_knn)
print("KNN Precision: ",precision_knn)
print("KNN Recall: ", recall_knn)
print("KNN accuracy score: ",knn.score(x_test,y_test))
print("____________________________")

# find best k value
import matplotlib.pyplot as plt
score_list=[]

for each in range(1,40):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,40),score_list)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
