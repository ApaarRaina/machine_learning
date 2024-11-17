import math
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris=load_iris()
X=iris.data
y=iris.target

X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

class AdaBoost:

    def __init__(self,no_of_stumps=5,trees=[]):
        self.no_of_stumps=no_of_stumps
        self.weights=np.full(X_train.shape[0],1/X_train.shape[0])
        self.trees=trees

    def error(self,y,y_predict):
        err=0
        y=np.array(y)
        y_predict=np.array(y_predict)
        for i in range(len(y)):
            if y[i]!=y_predict[i]:
                err+=self.weights[i]
        return err

    def performance(self,error):
        if error==0:
            return 0
        return 0.5*math.log((1-error)/error)

    def new_weights(self,ps,y,y_predict):
        y=np.array(y)
        y_predict=np.array(y_predict)
        if ps==0:
            return
        print("weights",self.weights)
        adjustment=np.where(y!=y_predict,np.exp(ps),np.exp(-ps))
        print(adjustment)
        self.weights=self.weights*adjustment

    def normalised_weights(self):
        sum=np.sum(self.weights)
        self.weights=self.weights/sum

    def ranges(self):
        higher=[]
        for i in range(1,len(self.weights)):
            higher=self.weights[i]+self.weights[i-1]
        lower=higher-self.weights
        return higher,lower

    def fit(self,no_of_stumps,X,y):
        for i in range(no_of_stumps):
            tree=DecisionTreeClassifier()
            tree.fit(X,y)
            self.trees.append(tree)
            y_predict=tree.predict(X)
            err=self.error(y,y_predict)
            ps=self.performance(err)
            self.new_weights(ps,y,y_predict)
            self.normalised_weights()
            random_array=np.random.rand(X.shape[0])
            higher=np.cumsum(self.weights)
            lower=np.roll(higher,1)
            lower[0]=0
            indexes=np.where((random_array>lower) & (random_array<=higher))
            X=X.iloc[indexes,:]
            y=y.iloc[indexes,:]

    def predict(self,X,y):
        trees=self.trees
        predict=[]
        for tree in trees:
            predict.append(tree.predict(X))
        final=np.apply_along_axis(lambda x:np.bincount(x).argmax(),axis=0,arr=predict)
        acc=accuracy_score(y,final)
        print(acc)



ada=AdaBoost()
ada.fit(5,X_train,y_train)
ada.predict(X_test,y_test)










