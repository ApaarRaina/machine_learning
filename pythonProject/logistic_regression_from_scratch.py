import pandas as pd
from sklearn.model_selection import train_test_split
import math
import  numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("spambase.data")

clean=data.iloc[:,[i for i in range(48,58)]]
train,test=train_test_split(data,test_size=0.1)

def sigmoid(line):
        np.clip(line,-500,500)
        return 1 / (1 + np.exp(-line))

def cost_function(clean, m, b):
    E = 0
    n = len(clean)
    X = clean.iloc[:, [0, 48, 55, 56]]
    y = clean.iloc[:, 57]
    line=np.dot(X,m)+b
    h = sigmoid(line)

    # Calculate the cost for this instance and add it to the total error
    E =np.mean(y*np.log(h)+(1-y)*np.log(1-h))

    return E



def gradient_decent(clean,m,b,L):
    n=len(clean)
    X=clean.iloc[:,[0,48,55,56]]
    y = clean.iloc[:, 57]

    line=np.dot(X,m) +b
    h= sigmoid(line)

    error=h-y
    m=m-L*(np.dot(X.T,error))/n
    b=b-L*np.mean(error)

    return m,b

def precision(test,m,b):
    n=len(test)
    tp,fp=0,0
    X = test.iloc[:, [0, 48, 55, 56]]
    y = test.iloc[:, 57]
    line=np.dot(X,m)+b
    y_predict=sigmoid(line)

    tp=np.sum((y_predict==1) & y==1)  #& is used for elementwise comparison in numpy
    fp=np.sum((y_predict==1) & y==0)

    return tp/(tp+fp)

epoch=1000
L=0.05
m=np.ones(4)
b=1
for i in range(epoch):
    if cost_function(train,m,b) <=1:
        break
    print(i)
    m,b=gradient_decent(train,m,b,L)


p=precision(test,m,b)
print(p)