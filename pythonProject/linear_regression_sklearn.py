import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

data=pd.read_csv("student-mat.csv",sep=";")

data= data[["G1","G2","G3","studytime","failures","absences"]]
predict="G3"

X=np.array(data.drop([predict],axis=1))
y=np.array(data[predict])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

linear=linear_model.LinearRegression()   #contains the line for linear regression

linear.fit(x_train,y_train)       #fits the line according to the data points
acc=linear.score(x_test,y_test)
print(acc)
