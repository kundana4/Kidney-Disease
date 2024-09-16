import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import missingno as msn

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r"kidney_disease.csv")
print(df)
print(df.isnull().sum())

msn.bar(df,color="red");
df.isnull()
df.duplicated().value_counts()
df['classification'].value_counts()
df['classification'].unique()
df[df["classification"]=="ckd\t"]
df["classification"]=df["classification"].replace("ckd\t","ckd",regex=True)

plt.figure(figsize=(17,7))
sns.countplot(data=df, x="classification")
plt.title("\nChronic Kidney Disease Distribution\n", fontsize=25)
plt.show();

df["age"].isnull().sum()
df["age"]=df["age"].fillna(df["age"].mean())
df.info()

# Here,we will fill all values of float64 datatype with the median and mode
numerical=[]
for col in df.columns:
    if df[col].dtype=="float64":
        numerical.append(col)
print(numerical)
for col in df.columns:
    if col in numerical:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Label Encoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
object_col = [col for col in df.columns if df[col].dtype == 'object']
for col in object_col:
    df[col] = le.fit_transform(df[col])

df.info()
print(df.columns)

# Visualizations
sns.boxplot(x=df['classification'], y=df['bu'])
plt.show();

sns.scatterplot(data=df,x="su",y="htn",hue='classification');
sns.stripplot(x=df["su"]);

sns.catplot(x="htn",y="su",data=df,kind="box");
plt.xlabel("Sugar",color="red")
plt.ylabel("Classification",color="red")
plt.title("Boxplot of Hypertension and Sugar",color="green");
plt.figure(figsize=(20,10))
plt.show()

sns.boxplot(data=df, x="ane", y="hemo", palette='seismic')
plt.xlabel("Hemeoglobin",color="red")
plt.ylabel("Aneamia",color="red")
plt.show()

plt.figure(figsize=(20,10))
sns.boxplot(data=df, y='hemo', x="pcc", hue="ane")
plt.xlabel("Hameoglobin",color="red")
plt.ylabel("Pus Cell Clumps",color="red")
plt.show()

print(df['classification'].shape)
df.groupby("classification").mean()
X=df[[ 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']]
y=df[['classification']]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=222)

print("Training Data ::-")
print("The shape of X training data is :-" ,X_train.shape)
print("The shape of y training data is :-" ,y_train.shape)

print("Testing Data ::-")
print("The shape of X testing data is :-" ,X_test.shape)
print("The shape of y testing data is :-" ,y_test.shape)


# Checking the correlated variables using heatmap(Pearson Correlation)
import seaborn as sns
plt.figure(figsize=(20,10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap)
plt.show();

# By using this function we can select correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.75)
print(len(set(corr_features)))

# Here,"Packed Cell Volume" and "Red Blood Cells (millions/cmm)" are the correlated features so we need to remove them. So,X needs to redefined again.
print(corr_features)
X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)

# Redifining the feature variable
X=df[['age', 'bp', 'sg', 'al', 'su',  'pcc', 'ba', 'bgr','bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad','appet', 'pe', 'ane']]
from sklearn.preprocessing import StandardScaler
sss=StandardScaler()
X=sss.fit_transform(X)
print(X.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=222)

print("Training Data ::-")
print("The shape of X training data is :-" ,X_train.shape)
print("The shape of y training data is :-" ,y_train.shape)

print("Testing Data ::-")
print("The shape of X testing data is :-" ,X_test.shape)
print("The shape of y testing data is :-" ,y_test.shape)

# Logistic Regression
# It is an supervised learning algorithm which is used for the solving classification problems.

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=200,random_state=222)
print(model)
model.fit(X_train,y_train)

# Prediction
y_predic=model.predict(X_test)
print(y_predic)
sns.countplot(y_predic);
print(model.predict_proba(X_test))

# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

print("Accuracy of the model is :  %3f " % accuracy_score(y_test,y_predic))
print(confusion_matrix(y_test,y_predic))
print(classification_report(y_test, y_predic))
