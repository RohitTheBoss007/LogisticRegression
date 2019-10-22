# Train a logistic regression classifier to predict whether flower is iris virginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
import numpy as np
iris=datasets.load_iris()

x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)

print(iris)

clf=LogisticRegression()

clf.fit(x,y)

example=clf.predict([[3.6]])
# print(example)