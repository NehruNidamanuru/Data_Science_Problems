import pandas as pd
import os
import numpy as np
from sklearn import preprocessing,decomposition,svm ,ensemble
from sklearn import naive_bayes,tree
from sklearn import model_selection,linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from mlxtend.classifier import StackingClassifier


os.chdir('/home/atchyuta/Problems/Hackerearth1.2')
train_data = pd.read_csv('train.csv')
train_data.shape
train_data.info()

train_data1=train_data[train_data['target']==0]
train_data1.shape
train_data2=train_data[train_data['target']==1]
train_data2.shape


train_data.describe()
test_data = pd.read_csv('test.csv')
test_data.shape
test_data['target']=''
total_data =pd.concat([train_data1,train_data2,test_data])
total_data.shape
print(train_data.groupby('target').size())

def get_features_missing_data(df, cutoff):
    total_missing = df.isnull().sum()
    n = df.shape[0]
    to_delete = total_missing[(total_missing/n) > cutoff ]
    return list(to_delete.index)

def filter_features(df, features):
    df.drop(features, axis=1, inplace=True)

missing_features = get_features_missing_data(total_data, 0)
filter_features(total_data, missing_features)

total_data.shape
total_data.info()
##total_data['cat_var_1'] = total_data['cat_var_1'].fillna(total_data['cat_var_1'].value_counts().index[0])
##total_data['cat_var_3'] = total_data['cat_var_3'].fillna(total_data['cat_var_3'].value_counts().index[0])
##total_data['cat_var_6'] = total_data['cat_var_6'].fillna(total_data['cat_var_6'].value_counts().index[0])
##total_data['cat_var_8'] = total_data['cat_var_8'].fillna(total_data['cat_var_8'].value_counts().index[0])


total_data1=total_data.drop(['transaction_id','target'],axis=1,inplace=False)
total_data1.shape

cont_data =total_data1.select_dtypes(include=['number']).columns
cat_data =total_data1.select_dtypes(exclude=['number']).columns
total_data2=pd.get_dummies(total_data1,columns=cat_data)
total_data2.shape
X_train=total_data2[0:train_data1.shape[0]]
X_train.shape
X_train.info()
import time
start_time = time.time()
oneclass = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
oneclass.fit(X_train)
print("--- %s seconds ---" % (time.time() - start_time))

train_test=total_data2[train_data1.shape[0]:train_data1.shape[0]+train_data2.shape[0]]
train_test.shape

fraud_pred = oneclass.predict(train_test)

unique, counts = np.unique(fraud_pred, return_counts=True)
print (np.asarray((unique, counts)).T)



X_test = total_data2[train_data1.shape[0]+train_data2.shape[0]:]
X_test.shape
X_test.info()

test_pred = frif.decision_function(X_test)
test_data['target'] =test_pred
test_data.to_csv('submission13iso0.csv', columns=['transaction_id','target'],index=False)

