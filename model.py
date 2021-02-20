import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import StratifiedKFold 
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import f1_score

from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 500)

#loading the dataset

data = pd.read_csv("Insurance Dataset.csv")

# Head of the dataset
data.head()

# Tail of the dataset
data.tail()

# Min/max of the dataset
data.describe()

#Information about the dataset
data.info()

# Display statistics in objects present in dataset
data.describe(include = np.object)

for column in data.columns:
     print("\n" + column)
     print(data[column].value_counts()) 

# Renaming the columns

data =  data.rename(columns = {'Emergency dept_yes/No': 'Emergency_dept','Home or self care,':'Home_or_self_care',
                                               'Mortality risk':'Mortality_risk','Hospital County':'Hospital_County','Hospital Id':'Hospital_Id'})

# View shape of data
data.shape

# Removing missing data
data.isnull().sum()

# Dropping columns with nan values
data = data.dropna(subset=['Hospital_Id','Mortality_risk','Area_Service','Hospital_County'],how = 'any')

#removing  duplicates
data.drop_duplicates(keep='first',inplace=True)

data.duplicated()

#Encoding the categorical variables using label encoder
 
def MultiLabelEncoder(columnlist,dataframe):
    for i in columnlist:

        labelencoder_X = LabelEncoder()
        dataframe[i] = labelencoder_X.fit_transform(dataframe[i])


columnlist = ['Area_Service','Hospital_County','Age','Gender','Days_spend_hsptl','Admission_type','Home_or_self_care','Surg_Description',
              'Emergency_dept','Cultural_group','ethnicity','apr_drg_description','Abortion']
MultiLabelEncoder(columnlist,data)

# Feature Engineering

X = data.loc[:, data.columns != 'Result']
y = data.loc[: , 'Result'].values

print(X)

print(y)

# Dropping columns
data.drop(['Hospital_Id', 'apr_drg_description', 'Abortion', 'Weight_baby'], axis=1, inplace=True)

# taking 5% sample data from whole dataset

sample_data = data.sample(frac=0.02)

# checking for result column unique values counts..

sample_data.Result.value_counts()

#Upsampling before splitting the data into train & test
# seperating '0', & '1' label 

sample_minority = sample_data.loc[sample_data['Result']==0]
sample_majority = sample_data.loc[sample_data['Result']==1]

sample_data_minority_upsampled = resample(sample_minority , replace=True, n_samples=12000, random_state=42)

sample_data_upsampled = pd.concat([sample_majority, sample_data_minority_upsampled ], ignore_index=True)

# Upsampling using sklearn resample
X = sample_data.drop(['Result'],axis=1).values
y = sample_data.Result.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)
X_train_df = pd.DataFrame(X_train, columns=sample_data.drop(['Result'],axis=1).columns)
y_train_df = pd.DataFrame(y_train, columns=['Result'])

df_train = pd.concat([X_train_df, y_train_df],axis=1)

df_train_majority = df_train.loc[df_train['Result']==1]
df_train_minority = df_train.loc[df_train['Result']==0]

df_train_minority_upsampled = resample(df_train_minority , replace=True, n_samples=9990, random_state=42)
after_split_upsampled = pd.concat([df_train_majority, df_train_minority_upsampled])

X_train_upsampled = after_split_upsampled.drop(['Result'],axis=1).values
y_train_upsampled = after_split_upsampled.Result.values

sample_RF15_after_split_upsampled = RandomForestClassifier(n_estimators=15, random_state=42)
sample_RF15_after_split_upsampled = sample_RF15_after_split_upsampled.fit(X_train_upsampled, y_train_upsampled)
y_pred_train = sample_RF15_after_split_upsampled.predict(X_train_upsampled)
y_pred = sample_RF15_after_split_upsampled.predict(X_test)
print(classification_report(y_train_upsampled, y_pred_train ))

# Building Model on whole dataset
data_minority = data.loc[data['Result']==0]
data_majority = data.loc[data['Result']==1]
data_minority_upsampled = resample(data_minority, replace=True, n_samples=624290, random_state=42)
data_upsampled = pd.concat([data_majority, data_minority_upsampled ], ignore_index=True)
X = data_upsampled.drop(['Result'],axis=1).values
y = data_upsampled.Result.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=30)


model_rf_upsampled = RandomForestClassifier(n_estimators=15, random_state=42)

model_rf_upsampled.fit(X_train, y_train)

y_pred_train = model_rf_upsampled.predict(X_train)
y_pred_test = model_rf_upsampled.predict(X_test)

print(classification_report(y_train, y_pred_train))

print(classification_report(y_test, y_pred_test))

# Building model
model = RandomForestClassifier(n_estimators=15, random_state=42)

model = model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Classification report
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
y_pred_whole = model.predict(sample_data.drop(['Result'],axis=1))

sns.heatmap(confusion_matrix(sample_data.Result.values, y_pred_whole), annot=True, fmt='.8g')
plt.show()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) 
lst_accu_stratified = [] 

X = data.drop(['Result'],axis=1).values
y = data.Result.values
   
for train_index, test_index in skf.split(X, y): 
    X_train_fold, X_test_fold = X[train_index], X[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index] 
    model_rf_upsampled.fit(X_train_fold, y_train_fold)
    lst_accu_stratified.append(f1_score(y_test_fold, model_rf_upsampled.predict(X_test_fold)))

print('List of possible F1 score:', lst_accu_stratified) 
print('\nMaximum F1 score That can be obtained from this model is:', 
      (np.round(max(lst_accu_stratified)*100, 2)))
print('\nMinimum F1 score:', 
      (np.round(min(lst_accu_stratified)*100,2))) 
print('\nOverall F1 score:', 
     (np.round(np.mean(lst_accu_stratified)*100,2)))
print('\nStandard Deviation is:', np.round(np.std(lst_accu_stratified)))


from joblib import dump, load
dump(model, 'model.joblib')














