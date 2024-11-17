import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data=pd.read_csv('spambase.data')
X=data.iloc[:,[0,48,55,56]]
Y=data.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=20)

model=LogisticRegression()
model.fit(X_train,Y_train)
predict=model.predict(X_test)

accuracy=metrics.accuracy_score(Y_test,predict)
print(accuracy)
