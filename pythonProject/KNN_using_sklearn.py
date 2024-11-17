import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('spambase.data')

X=data.iloc[:,[0,1,2,3,4,5,48,49,50,51,53,54,55,56]]
y=data.iloc[:,57]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)

print(accuracy_score(y_test,y_predict))


