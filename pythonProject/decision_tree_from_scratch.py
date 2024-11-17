import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X=pd.read_csv('spambase.data')

y=X.iloc[:,-1]
X=X.iloc[:,:-1]
X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
s=len(y)

def entropy(y):
    total=len(y)
    unique,counts=np.unique(y.iloc[:,0],return_counts=True)
    p=counts/total
    log=-np.log(p)
    ent=np.sum(log*p)
    return ent

def I_G(y,ent,s):
    sv=len(y)
    unique, counts = np.unique(y.iloc[:, 0],return_counts=True)
    p = counts / sv
    log = -np.log(p)
    ent_sv=np.sum(log*p)
    gain=ent-(sv/s)*ent_sv

    return gain


def split_left(feature,X,y):
    threshold=feature.mean()
    left_index=X[feature<threshold].index

    X_left=X.loc[left_index]
    y_left= y.loc[left_index]

    return X_left,y_left


def split_right(feature, X, y):
    threshold = feature.mean()
    right_index = X[feature >= threshold].index

    X_right = X.loc[right_index]
    y_right = y.loc[right_index]

    return X_right, y_right


ent=entropy(y)
class Node:


    def __init__(self,left=None,right=None,feature=None,threshold=None,value=None):
        self.left=left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value


class Tree:

   def __init__(self):
       self.max_depth = 0

   def fit(self,X,y):
       self.root=self.build(X,y)

   def build(self,X,y,depth=0):

       if depth==10:
           if y.shape[0]==0:
               leaf_value=1
               return Node(value=leaf_value)
           unique_values,counts=np.unique(y,return_counts=True)
           value_index=np.argmax(counts)
           leaf_value=unique_values[value_index]
           return Node(value=leaf_value)

       if len(np.unique(y))<=1:
           if y.shape[0]==0:
               leaf_value=1
               return Node(value=leaf_value)
           leaf_value=y.iloc[0,0]
           return Node(value=leaf_value)

       l=[]
       for i in X:
           feature=X[i]
           s=len(y)
           X_left,y_left=split_left(feature,X,y)
           X_right, y_right = split_right(feature, X, y)
           sv_left = len(y_left)
           sv_right = len(y_right)
           if s==0:
               gain=0
           else:
              gain=(sv_left/s)*entropy(y_left)+(sv_right/s)*entropy(y_right)
              ent=entropy(y)
              gain=ent-gain
           l.append(gain)

       max_gain=max(l)
       index=l.index(max_gain)
       best_feature=X.iloc[:,index]
       threshold=best_feature.mean()
       X_left,y_left=split_left(best_feature,X,y)
       X_right,y_right=split_right(best_feature,X,y)
       left_tree=self.build(X_left,y_left,depth+1)
       right_tree=self.build(X_right,y_right,depth+1)
       return Node(left=left_tree,right=right_tree,feature=X.columns[index],threshold=threshold)

   def predict_single(self,x,root):
       if root.value is not None:
           return root.value

       if x[root.feature]<root.threshold:
           return self.predict_single(x,root.left)
       else:
           return self.predict_single(x,root.right)

   def predict(self, X):
       # Predict for each instance in the DataFrame X
       predictions = [self.predict_single(x, self.root) for _, x in X.iterrows()]
       return np.array(predictions)


tree=Tree()
tree.fit(X_train,y_train)
y_predict=tree.predict(X_test)

acc=accuracy_score(y_test,y_predict)

print(acc)

















       





















