import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn import tree,model_selection
import io
import pydot

os.chdir('/home/atchyuta/Problems/Classification_Titanic')
Tita_train=pd.read_csv('train.csv')
Tita_train.shape
Tita_train.info()
Tita_train.describe()
Tita_test=pd.read_csv('test.csv')
Tita_test.shape
Tita_test.info()
Tita_test.describe()

Tita_train.groupby(['Sex','Survived']).size()

##Univariate

#Categorical columns:Statistical EDA
pd.crosstab(index=Tita_train['Survived'],columns='count')
pd.crosstab(index=Tita_train['Sex'],columns='count')
pd.crosstab(index=Tita_train['Pclass'],columns='count')
pd.crosstab(index=Tita_train['Embarked'],columns='count')

#Categorical columns Visual EDA
sns.countplot(x='Survived',data=Tita_train)
sns.countplot(x='Sex',data=Tita_train)
sns.countplot(x='Pclass',data=Tita_train)
sns.countplot(x='Embarked',data=Tita_train)

##Cintinuious column Statistical EDA
Tita_train['Fare'].describe()
#Continuious columns visual EDA
sns.boxplot(x='Fare',data=Tita_train)
sns.distplot(Tita_train['Fare'])
Tita_train['logfare']=np.log(Tita_train['Fare'])
sns.distplot(Tita_train['logfare'],bins=100,kde=False)
sns.boxplot(x='logfare',data=Tita_train)
sns.distplot(Tita_train['Age'])





##Bivariate
#Categorical columns:Statistical EDA
pd.crosstab(index=Tita_train['Survived'],columns=Tita_train['Sex'])
pd.crosstab(index=Tita_train['Survived'],columns=Tita_train['Pclass'])
pd.crosstab(index=Tita_train['Survived'],columns=Tita_train['Embarked'],margins=True)
#Categorical columns Visual EDA
sns.factorplot(x='Sex',hue='Survived',data=Tita_train,kind='count',size=6)
sns.factorplot(x='Embarked',hue='Survived',data=Tita_train,kind='count',size=6)
sns.factorplot(x='Pclass',hue='Survived',data=Tita_train,kind='count',size=6)
##Categorical to continuious Visual EDA
sns.FacetGrid(Tita_train,row='Survived',size=8).map(sns.kdeplot,'Fare').add_legend()
sns.FacetGrid(Tita_train,row='Survived',size=8).map(sns.boxplot,'Fare').add_legend()
sns.FacetGrid(Tita_train,row='Survived',size=8).map(sns.boxplot,'Age').add_legend()

##multivariate
sns.FacetGrid(Tita_train,row='Survived',col='Fare',size=8).map(sns.kdeplot,'Age').add_legend()

Tita_train['Embarked'] = Tita_train['Embarked'].fillna(Tita_train['Embarked'].value_counts().index[0])
Tita_train.info()

Tita_train1=pd.get_dummies(Tita_train,columns=['Sex'])
Tita_train1.shape
Tita_train1.info()


#convert categorical columns to numeric type
ordinal_features1 = ['Embarked']
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "C": 1, "S": 2, "Q": 3}
for feature in ordinal_features1:
    null_idx =  Tita_train1[feature].isnull()
    Tita_train1.loc[null_idx, feature] = None 
    Tita_train1[feature] = Tita_train1[feature].map(quality_dict)

pd.crosstab(index=Tita_train['Embarked'],columns='count')


X_train=Tita_train1[['Sex_female','Sex_male','Embarked','Pclass']]
y_train=Tita_train['Survived']
tree_estimator=tree.DecisionTreeClassifier()
print(model_selection.cross_val_score(tree_estimator, X_train, y_train, cv= 10).mean())
tree_estimator.fit(X_train, y_train)
print(tree_estimator.score(X_train, y_train))

##Visualize Decision Tree

dot_data=io.StringIO()
tree.export_graphviz(tree_estimator,out_file=dot_data,feature_names=X_train.columns)
graph=pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf('decission_tree.pdf')

##Test
Tita_test['Fare'] = Tita_test['Fare'].fillna(Tita_test['Fare'].value_counts().index[0])

Tita_test['Embarked'] = Tita_test['Embarked'].fillna(Tita_test['Embarked'].value_counts().index[0])
Tita_test.info()

Tita_test1=pd.get_dummies(Tita_test,columns=['Sex'])
Tita_test1.shape
Tita_test1.info()


#convert categorical columns to numeric type
ordinal_features1 = ['Embarked']
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "C": 1, "S": 2, "Q": 3}
for feature in ordinal_features1:
    null_idx =  Tita_test1[feature].isnull()
    Tita_test1.loc[null_idx, feature] = None 
    Tita_test1[feature] = Tita_test1[feature].map(quality_dict)

pd.crosstab(index=Tita_test1['Embarked'],columns='count')

X_test=Tita_test1[['Sex_female','Sex_male','Embarked','Pclass']]


X_test=Tita_test1.drop(['Age','Name','PassengerId','Ticket','Cabin'],axis=1,inplace=False)
X_test.info()
Tita_test['Survived']=tree_estimator.predict(X_test)
Tita_test.to_csv('submission1.csv', columns=['PassengerId','Survived'],index=False)

