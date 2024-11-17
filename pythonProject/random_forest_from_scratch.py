import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

iris=load_iris()
X=iris.data
y=iris.target
X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
split_X=[]
split_y=[]
trees=[]

class RandomForest:

    def __init__(self,no_of_splits=10):
        self.no_of_splits=10

    def fit(self,X,y):
        split_X=np.array_split(X,self.no_of_splits)
        split_y = np.array_split(y, self.no_of_splits)
        for i in range(len(split_X)):
            tree=DecisionTreeClassifier()
            tree.fit(split_X[i],split_y[i])
            trees.append(tree)

    def predict(self,X,y):
        y_predict=[]
        for _,row in X.iterrows():
            pred=[]
            row=row.values.reshape(1,-1)
            for i in trees:
                pred.append(i.predict(row)[0])
            y_predict.append(np.bincount(pred).argmax())

        y_predict=pd.DataFrame(y_predict)
        acc=accuracy_score(y,y_predict)
        print(acc)

rf=RandomForest()
rf.fit(X_train,y_train)
rf.predict(X_test,y_test)

