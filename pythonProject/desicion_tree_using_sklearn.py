import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

X=pd.read_csv('spambase.data')

y=X.iloc[:,-1]
X=X.iloc[:,:-1]
X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

y_predict=tree.predict(X_test)
print(classification_report(y_test,y_predict))




