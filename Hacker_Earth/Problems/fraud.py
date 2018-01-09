import pandas as pd
import os
from sklearn import preprocessing,decomposition,svm ,ensemble
from sklearn import naive_bayes,tree
from sklearn import model_selection,linear_model
from sklearn.svm import SVC


os.chdir('/home/atchyuta/Problems/Hackerearth1.2')
train_data = pd.read_csv('train.csv')
train_data.shape
train_data.info()
train_data.describe()
test_data = pd.read_csv('test.csv')
test_data.shape
test_data['target']=0
total_data =pd.concat([train_data,test_data])
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
X_train=total_data2[0:train_data.shape[0]]
X_train.shape
X_train.info()
y_train=train_data['target']

dt_estimator = tree.DecisionTreeClassifier(random_state=20)
print(model_selection.cross_val_score(dt_estimator,X_train,y_train,cv=10).mean())
dt_estimator.fit(X_train,y_train)
X_test = total_data2[train_data.shape[0]:]


import time
start_time = time.time()
clf=SVC()
clf.fit(X_train,y_train)
print("--- %s seconds ---" % (time.time() - start_time))


X_test.shape
X_test.info()
test_data['target'] = dt_estimator.predict(X_test)
test_data.to_csv('submission13dt2.csv', columns=['transaction_id','target'],index=False)





dt_grid ={'max_features':[7], 'max_depth':[5]}
grid_dt_estimator=model_selection.GridSearchCV(dt_estimator, dt_grid,scoring='roc_auc',cv=10,n_jobs=1)
grid_dt_estimator.fit(X_train,y_train)



pca = decomposition.PCA()
pca.fit(X_train)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

pca = decomposition.PCA(150)
pca.fit(X_train)
X_train1 = pca.transform(X_train)
X_train1.info()
y_train=train_data['target']


lr_estimator = linear_model.LogisticRegression(random_state=21)
lr_grid = {'penalty':['l1','l2'], 'max_iter':[100], 'C':[0.7,1]}
grid_lr_estimator = model_selection.GridSearchCV(lr_estimator, lr_grid,scoring='roc_auc',cv=10,n_jobs=1)
grid_lr_estimator.fit(X_train1, y_train)
print(grid_lr_estimator.grid_scores_)
print(grid_lr_estimator.best_score_)
print(grid_lr_estimator.best_params_)
print(grid_lr_estimator.score(X_train1, y_train))
final_model = grid_lr_estimator.best_estimator_
final_model.coef_
final_model.intercept_

nb_estimator = naive_bayes.GaussianNB()
mean_cv_score = model_selection.cross_val_score(nb_estimator, X_train,y_train,scoring='roc_auc',cv=10).mean()
nb_estimator.fit(X_train,y_train)
nb_estimator.class_prior_
nb_estimator.sigma_
nb_estimator.theta_

X_test = total_data2[train_data.shape[0]:]




X_test.shape
X_test.info()
test_data['target'] = nb_estimator.predict_proba(X_test)
test_data.to_csv('submission13nb0.csv', columns=['transaction_id','target'],index=False)


lsvm_estimator = svm.LinearSVC(random_state=2017)
lsvm_grid = {'C':[0.1,0.2,0.5,1] }
grid_lsvm_estimator = model_selection.GridSearchCV(lsvm_estimator, lsvm_grid,scoring="roc_auc", cv=10, n_jobs=1)
grid_lsvm_estimator.fit(X_train1, y_train)
print(grid_lsvm_estimator.grid_scores_)
print(grid_lsvm_estimator.best_score_)
print(grid_lsvm_estimator.best_params_)
print(grid_lsvm_estimator.score(X_train1, y_train))
final_model = grid_lsvm_estimator.best_estimator_
final_model.coef_
final_model.intercept_


gbm_estimator = GradientBoostingClassifier(random_state=2017)
gbm_grid = {'n_estimators':[50, 100], 'max_depth':[3,4,5], 'learning_rate':[0.001,0.01,0.2,0.3]}
grid_gbm_estimator = model_selection.GridSearchCV(gbm_estimator, gbm_grid,scoring="roc_auc",cv=10,n_jobs=1)
grid_gbm_estimator.fit(X_train,y_train)
print(grid_gbm_estimator.grid_scores_)
print(grid_gbm_estimator.best_score_)
print(grid_gbm_estimator.best_params_)
print(grid_gbm_estimator.score(X_train, y_train))

isf=ensemble.IsolationForest(random_state=2017)


X_test = total_data2[train_data.shape[0]:]
pca = decomposition.PCA()
pca.fit(X_test)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

pca = decomposition.PCA(150)
pca.fit(X_test)
X_test1 = pca.transform(X_test)
X_test.shape
X_test.info()
test_data['target'] = dt_estimator.predict_proba(X_test1)

test_data['target'] = nb_estimator.predict_proba(X_test1)

test_data.to_csv('submission13lr0.csv', columns=['transaction_id','target'],index=False)




##Stacking
##with stack model almost getting 96

dt_estimator = tree.DecisionTreeClassifier(random_state=20)
nb_estimator = naive_bayes.GaussianNB()

stage1_models = [nb_estimator, dt_estimator]
stage2_model = tree.DecisionTreeClassifier(random_state=20)

stacked_model = StackingClassifier(classifiers=stage1_models,meta_classifier=stage2_model)
stack_estimator=model_selection.cross_val_score(stacked_model, X_train, y_train, 
                                              cv=3, scoring='roc_auc')





      
stack_estimator.fit(X_train, y_train)
print(stacked_model.grid_scores_)
print(stacked_model.best_params_)
print(stacked_model.best_score_)
print(stacked_model.score(X_train1,y_train))

stacked_model.fit(X_train1, y_train)
stacked_model.predict(X_train)












X_test = total_data2[train_data.shape[0]:]
pca = decomposition.PCA()
pca.fit(X_test)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

pca = decomposition.PCA(150)
pca.fit(X_test)
X_test1 = pca.transform(X_test)
X_test.shape
X_test.info()
test_data['target'] = stacked_model.predict(X_test)

test_data.to_csv('submission13stack0.csv', columns=['transaction_id','target'],index=False)

