import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree

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

# plt.figure(figsize=(17,7))
# sns.countplot(data=df, x="classification")
# plt.title("\nChronic Kidney Disease Distribution\n", fontsize=25)
# plt.show();

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

# Build the initial decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Print the initial accuracy
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
print("Initial Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Initial Test Accuracy:", accuracy_score(y_test, y_pred_test))

# Pruning the tree
# Use cost complexity pruning (minimal cost-complexity pruning)
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Create a list of DecisionTreeClassifier models for each ccp_alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Evaluate each pruned model
for clf in clfs:
    y_pred_test = clf.predict(X_test)
    print(f"Test Accuracy for ccp_alpha={clf.ccp_alpha}: {accuracy_score(y_test, y_pred_test)}")

# Choose the best model (e.g., one with highest accuracy or lowest complexity)
best_clf = max(clfs, key=lambda clf: accuracy_score(y_test, clf.predict(X_test)))

# Print the final accuracy
y_pred_best = best_clf.predict(X_test)
print("Best Test Accuracy:", accuracy_score(y_test, y_pred_best))

# Plot the best pruned tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns)
plt.show()
