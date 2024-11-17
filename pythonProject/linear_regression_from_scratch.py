import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data1=pd.read_csv('student-mat.csv',sep=';')
clean=data1.loc[:,['studytime','G1','G2','G3']]
train,test=train_test_split(clean,test_size=0.2)

def gradient_descent(m1_now,m2_now,m3_now,b_now,points,L):
    m1_gradient=0
    m2_gradient = 0
    m3_gradient = 0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x1=points.iloc[i].studytime
        x2=points.iloc[i].G1
        x3=points.iloc[i].G2
        y=points.iloc[i].G3

        m1_gradient+=-(2/n)*x1*(y-(m1_now*x1+m2_now*x2+m3_now*x3+b_now))
        m2_gradient += -(2 / n) * x2 * (y - (m1_now * x1 + m2_now * x2 + m3_now * x3 + b_now))
        m3_gradient += -(2 / n) * x3 * (y - (m1_now * x1 + m2_now * x2 + m3_now * x3 + b_now))

        b_gradient+=-(2/n)*(y-(m1_now*x1+m2_now*x2+m3_now*x3+b_now))

    m1=m1_now-L*m1_gradient
    m2 = m2_now - L * m2_gradient
    m3 = m3_now - L * m3_gradient
    b=b_now-L*b_gradient

    return m1,m2,m3,b


def accuracy(m1,m2,m3,b,test):

    mean=0
    for i in range(len(test)):
        y = test.iloc[i].G3
        mean+=y
    mean/=len(test)

    var_mean=0
    for i in range(len(test)):
        y=test.iloc[i].G3
        var_mean+=(mean-y)**2
    var_mean/=len(test)

    var_line=0
    for i in range(len(test)):
        x1=test.iloc[i].studytime
        x2=test.iloc[i].G1
        x3=test.iloc[i].G2
        y=test.iloc[i].G3
        var_line+=((m1 * x1 + m2 * x2 + m3 * x3 + b)-y)**2
    var_line/=len(test)

    r_square=(var_mean-var_line)/var_mean
    return r_square

m1=0
m2=0
m3=0
b=0
L=0.0001
epochs=300

for i in range(epochs):
    m1,m2,m3,b= gradient_descent(m1,m2,m3,b,train,L)


print(m1,m2,m3,b)
r_test=accuracy(m1,m2,m3,b,test)
r_train=accuracy(m1,m2,m3,b,train)
print(r_test)
print(r_train)




