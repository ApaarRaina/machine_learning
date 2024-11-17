import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

data=pd.read_csv('spambase.data')

X=data.iloc[:,[0,1,2,3,4,5,48,49,50,51,53,54,55,56]]
y=data.iloc[:,57]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

naive_bayes=GaussianNB()
naive_bayes.fit(X_train,y_train)

y_predict=naive_bayes.predict(X_test)

precision=classification_report(y_test,y_predict)
print(precision)

