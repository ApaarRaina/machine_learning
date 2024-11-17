import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("spambase.data")
X=data.iloc[:,[0,1,2,3,4,5,48,49,50,51,53,54,55,56]]
y=data.iloc[:,57]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

def K_distance(X_train,y_train,x,k):
    train=np.array(X_train)
    test=np.array(x)

    sq=np.square(train-test)
    sq=np.sum(sq,axis=1,dtype=np.int32)
    sq=np.sqrt(sq)

    sq=np.column_stack((sq,y_train))
    dist=np.sort(sq)[:k]
    dist=pd.DataFrame(dist)

    return dist.iloc[0,:]

def KNN(X_train,y_train,X_test):

    k=10
    y_predict=[]
    for i in range(len(X_test)):
        x=X_test.iloc[i,:]
        dist=K_distance(X_train,y_train,x,k)
        y_predict.append(np.bincount(dist).argmax())

    return y_predict


y_predict=KNN(X_train,y_train,X_test)
print(accuracy_score(y_test,y_predict))












