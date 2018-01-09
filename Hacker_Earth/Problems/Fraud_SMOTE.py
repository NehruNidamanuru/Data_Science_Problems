import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes,tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc

os.chdir('/home/atchyuta/Problems/Hackerearth1.2')
train_data = pd.read_csv('train.csv')
train_data.shape
train_data.info()
train_data.iloc[0]
train_data.target.value_counts()

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


total_data1=total_data.drop(['transaction_id','target'],axis=1,inplace=False)
total_data1.shape

cont_data =total_data1.select_dtypes(include=['number']).columns
cat_data =total_data1.select_dtypes(exclude=['number']).columns
total_data2=pd.get_dummies(total_data1,columns=cat_data)
total_data2.shape

final_train=total_data2[0:train_data.shape[0]]
final_train.shape
final_train.info()
final_train['target']=train_data['target']





training_features, test_features, \
training_target, test_target, = train_test_split(final_train.drop(['target'], axis=1),
                                               final_train['target'],
                                               test_size = .1,
                                               random_state=12)

training_features.shape
test_features.shape
training_target.shape
test_target.shape

sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(training_features, training_target)
len(y_res[y_res==1])
print (training_target.value_counts(), np.bincount(y_res))

x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(x_res,
                                                    y_res,
                                                    test_size = .1,
                                                    random_state=12)

clf_rf = RandomForestClassifier(random_state=12)
rf_grid={'n_estimators':[25]}
rf_grid_estimator=model_selection.GridSearchCV(clf_rf,rf_grid,scoring='roc_auc',cv=10,n_jobs=1)

rf_grid_estimator.fit(x_train_res, y_train_res)
rf_grid_estimator.score(x_val_res, y_val_res)

recall_score(y_val_res, rf_grid_estimator.predict(x_val_res))

print(rf_grid_estimator.score(test_features, test_target))
print(recall_score(test_target, rf_grid_estimator.predict(test_features)))


final_test=total_data2[train_data.shape[0]:]
final_test.shape
final_test.info()
test_data['target']=rf_grid_estimator.predict_proba(final_test)
test_data.to_csv('submission16smote0.csv', columns=['transaction_id','target'],index=False)


x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)

print('Validation Results')
print (clf_rf.score(x_val, y_val))
print (recall_score(y_val, clf_rf.predict(x_val)))
print ('\nTest Results')
print (clf_rf.score(test_features, test_target))
print (recall_score(test_target, clf_rf.predict(test_features)))

actual = test_target
predictions=clf_rf.predict_proba(test_features)
predictions.shape

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')