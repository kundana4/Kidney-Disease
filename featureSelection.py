import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# Load the CKD dataset

df=pd.read_csv(r"kidney_disease.csv")
print(df)
print(df.isnull().sum())

# msn.bar(df,color="red");
df.isnull()
df.duplicated().value_counts()
df['classification'].value_counts()
df['classification'].unique()
df[df["classification"]=="ckd\t"]
df["classification"]=df["classification"].replace("ckd\t","ckd",regex=True)

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

print(df['classification'].shape)
df.groupby("classification").mean()
X=df[[ 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']]
y=df[['classification']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Univariate Feature Selection
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train_scaled, y_train)
scores = selector.scores_

# Print feature scores
print("Univariate Feature Selection Scores:")
for feature, score in zip(X.columns, scores):
    print(f"{feature}: {score}")

# Recursive Feature Elimination (RFE)
# Using Logistic Regression for RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=5)  # Choose number of features to select
rfe.fit(X_train_scaled, y_train)

print("\nRFE Selected Features:")
for feature, support in zip(X.columns, rfe.support_):
    if support:
        print(feature)

# Feature Importances from Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train_scaled, y_train)
importances = forest.feature_importances_

# Print feature importances
print("\nFeature Importances from Random Forest:")
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")

# Evaluate feature selection with selected features
# Using RFE-selected features for evaluation
X_train_rfe = X_train_scaled[:, rfe.support_]
X_test_rfe = X_test_scaled[:, rfe.support_]

model.fit(X_train_rfe, y_train)
y_pred_test = model.predict(X_test_rfe)

print("\nRFE Model Evaluation:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")