import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

iris=load_iris()
X=iris.data
y=iris.target
X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
y_predict=ada.predict(X_test)
acc=accuracy_score(y_test,y_predict)
print(acc)




