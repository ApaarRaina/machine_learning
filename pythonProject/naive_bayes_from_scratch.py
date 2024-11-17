import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data=pd.read_csv('spambase.data')
data=data.iloc[:,[0,1,2,3,4,5,48,49,50,51,53,54,55,56,57]]
train,test=train_test_split(data,test_size=0.1)
y_test=test.iloc[:,14]

yes_train=data[data.iloc[:,14]!=0]
no_train=data[data.iloc[:,14]==0]

def yes_probability(yes_train,X):
    cnt=0
    yes=len(yes_train[yes_train.iloc[:,14]==1])/len(yes_train.iloc[:,14])
    total_p=yes
    for i in X:
        y=yes_train.iloc[:,cnt]
        x=yes_train[yes_train.iloc[:,cnt]==i]
        n=len(x)
        p=n/len(y)
        total_p*=p
        cnt+=1
    return total_p

def no_probability(no_train,X):
    cnt=0
    no=len(no_train[no_train.iloc[:,14]==1])/len(no_train.iloc[:,14])
    total_p=no
    for i in X:
        y=no_train.iloc[:,cnt]
        x=no_train[no_train.iloc[:,cnt]==i]
        n=len(x)
        p=n/len(y)
        total_p*=p
        cnt+=1
    return total_p

def standardise(yes,no):
    if yes==0:
        return 0
    return yes/(yes+no)


def predict(test):
    y_predict = []

    for i in range(len(test)):
        y_true = test.iloc[i, 14]  # True label
        X = test.iloc[i, :-1]  # Features

        # Calculate probabilities
        yes_p = yes_probability(yes_train, X)
        no_p = no_probability(no_train, X)

        # Standardize probabilities
        yes_p = standardise(yes_p, no_p)
        no_p = standardise(no_p, yes_p)

        # Determine predicted label
        output = 1 if yes_p >= 0.5 else 0
        y_predict.append(output)

    # Convert y_predict to a DataFrame
    y_predict_df = pd.DataFrame(y_predict, columns=["Prediction"])

    return y_predict_df

y_predict=predict(test)
print(classification_report(y_test,y_predict))




