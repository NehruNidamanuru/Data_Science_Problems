import pandas as pd
import os
import seaborn as sns
import math
from sklearn import tree,model_selection,metrics,ensemble
from mlxtend.regressor import StackingRegressor


##With out doing anything on data like to fill missing data (just remove the columns which are having missing data)

os.chdir('/home/atchyuta/Problems/Hackerearth1')
train_data = pd.read_csv('train.csv')
train_data.shape
train_data.info()
train_data.describe()
test_data = pd.read_csv('test.csv')
test_data['return']=0
test_data.shape
test_data.info()
total_data=pd.concat([train_data,test_data])
total_data.shape
total_data.info()
##total_missing = total_data['libor_rate'].isnull().sum()
##total_data.loc[total_data['libor_rate'].isnull()==True,'libor_rate']=train_data['libor_rate'].mean()
##total_missing = total_data['sold'].isnull().sum()
##total_data.loc[total_data['sold'].isnull()==True,'sold']=train_data['sold'].mean()
##total_missing = total_data['bought'].isnull().sum()
##total_data.loc[total_data['bought'].isnull()==True,'bought']=train_data['bought'].mean()
#total_data['bought']=pd.to_numeric(train_data['bought'],errors='coerce')
#total_data['sold']=pd.to_numeric(train_data['sold'],errors='coerce')
total_data.info()

total_data1=pd.get_dummies(total_data,columns=['pf_category','office_id','currency','country_code','type'])
total_data1.shape
total_data1.info()
sns.distplot(train_data['return'])






X_train =total_data1[0:train_data.shape[0]]
X_train.shape
X_train.info()

X=X_train.select_dtypes(include=['number']).columns
X_train1=X_train[X]
X_train1=X_train1.drop(['bought','sold','libor_rate','return'],axis=1,inplace=False)
X_train1.shape
X_train1.info()
y_train = train_data['return']
y_train.shape

## with this getting 99.4
tree_estimator = tree.DecisionTreeRegressor()
print(model_selection.cross_val_score(tree_estimator,X_train1,y_train,scoring='adjusted_rand_score',cv=10).mean())
print(model_selection.cross_val_score(tree_estimator,X_train1,y_train,scoring='r2',cv=10).mean())

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred))

print(model_selection.cross_val_score(tree_estimator,X_train1,y_train,scoring=metrics.make_scorer(rmse),cv=10).mean())

tree_estimator.fit(X_train1,y_train)


features = X_train1.columns
importances = tree_estimator.feature_importances_
important_features = pd.DataFrame({'feature':features, 'importance': importances})


X_test=total_data1[train_data.shape[0]:]
X_test.shape
X1=X_test.select_dtypes(include=['number']).columns
X_test1=X_test[X1]
X_test1.shape
X_test1.info()
X_test1=X_test1.drop(['bought','sold','libor_rate','return'],axis=1,inplace=False)

X_test1.shape
test_data['return']=tree_estimator.predict(X_test1)

test_data.to_csv('sub12Tree0.csv',columns=['portfolio_id','return'],index=False)


##Stacking
##with stack model almost getting 99.2

dt_estimator = tree.DecisionTreeRegressor(random_state=100, max_depth=5)
rf_estimator = ensemble.RandomForestRegressor(random_state=100, n_estimators=100, max_features=3)
gbm_estimator = ensemble.GradientBoostingRegressor(random_state=100, n_estimators=100, max_features=3, max_depth=5,
                            learning_rate=0.05)
stage1_models = [dt_estimator, rf_estimator, gbm_estimator]
stage2_model = tree.DecisionTreeRegressor(random_state=100)

stacked_model = StackingRegressor(regressors=stage1_models,meta_regressor=stage2_model)
      
stacked_model.fit(X_train1, y_train)
print(stacked_model.grid_scores_)
print(stacked_model.best_params_)
print(stacked_model.best_score_)
print(stacked_model.score(X_train1,y_train))

stacked_model.fit(X_train1, y_train)
stacked_model.predict(X_train1)












X_test=total_data1[train_data.shape[0]:]
X_test.shape
X1=X_test.select_dtypes(include=['number']).columns
X_test1=X_test[X1]
X_test1.shape
X_test1.info()
X_test1=X_test1.drop(['bought','sold','libor_rate','return'],axis=1,inplace=False)

X_test1.shape
test_data['return']=stacked_model.predict(X_test1)
test_data.to_csv('sub12stack1.csv',columns=['portfolio_id','return'],index=False)



##Extratrees
et_estimator = ensemble.ExtraTreesRegressor(random_state=2017)
et_grid = {'n_estimators':[6], 'max_features':[6,7,8], 'max_depth':[3,4,5]}
grid_et_estimator = model_selection.GridSearchCV(et_estimator, et_grid, cv=10,n_jobs=1)
grid_et_estimator.fit(X_train1, y_train)
print(grid_et_estimator.grid_scores_)
print(grid_et_estimator.best_score_)
print(grid_et_estimator.best_params_)
print(grid_et_estimator.score(X_train1, y_train))

features = X_train1.columns
importances = grid_et_estimator.best_estimator_.feature_importances_
imp_fet = pd.DataFrame({'feature':features, 'importance': importances})


##Bagging
bt_estimator=ensemble.BaggingRegressor(base_estimator=tree_estimator,oob_score=True,random_state=2017)
bt_grid={'n_estimators':[6],'base_estimator__max_depth':[3,4,5],'base_estimator__max_features':[6,7,8]}
grid_bt_estimator=model_selection.GridSearchCV(bt_estimator,bt_grid,cv=10,n_jobs=1)
grid_bt_estimator.fit(X_train1, y_train)
print(grid_bt_estimator.grid_scores_)
print(grid_bt_estimator.best_score_)
print(grid_bt_estimator.best_params_)
print(grid_bt_estimator.score(X_train1, y_train))   




##Adaboost
ada_estimator=ensemble.AdaBoostRegressor(base_estimator=tree_estimator,random_state=2017)
ada_grid={'n_estimators':[100],'learning_rate':[0.01,0.02,0.09,0.1,0.3,0.7,1.0],'base_estimator__max_depth':[5]}
grid_ada_estimator=model_selection.GridSearchCV(ada_estimator,ada_grid,cv=10,n_jobs=1)
grid_ada_estimator.fit(X_train1,y_train)
print(grid_ada_estimator.grid_scores_)
print(grid_ada_estimator.best_params_)
print(grid_ada_estimator.best_score_)
print(grid_ada_estimator.score(X_train1,y_train))

features=X_train1.columns
importances=grid_ada_estimator.best_estimator_.feature_importances_
imp_fet1=pd.DataFrame({"feature":features,"importance":importances})

##Gbm
gbm_estimator=ensemble.GradientBoostingRegressor(random_state=2017)
gbm_grid={'n_estimators':[100,150],'learning_rate':[0.01,0.02,0.09,0.1,0.3,0.7,1.0],'max_depth':[5,7]}
grid_gbm_estimator=model_selection.GridSearchCV(gbm_estimator,gbm_grid,cv=10,n_jobs=1)
grid_gbm_estimator.fit(X_train1,y_train)
print(grid_gbm_estimator.grid_scores_)
print(grid_gbm_estimator.best_params_)
print(grid_gbm_estimator.best_score_)
print(grid_gbm_estimator.score(X_train1,y_train))

features=X_train1.columns
importances=grid_ada_estimator.best_estimator_.feature_importances_
imp_fet2=pd.DataFrame({"feature":features,"importance":importances})




