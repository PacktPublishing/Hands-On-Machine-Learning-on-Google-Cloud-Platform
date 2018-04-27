from sklearn import linear_model, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from pandas_ml import ConfusionMatrix
import numpy as np
import pandas as pd

BC = datasets.load_breast_cancer()
print(BC.data.shape)
print(BC.target.shape)
X = BC.data
Y = BC.target

Xdata=pd.DataFrame(X)
print(Xdata.describe())

X = (X - np.min(X, 0)) / (np.max(X, 0) - np.min(X, 0))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

RbmModel = BernoulliRBM(random_state=0, verbose=True)
FitRbmModel = RbmModel.fit_transform(X_train, Y_train)
LogModel = linear_model.LogisticRegression()
LogModel.coef_ = FitRbmModel
Classifier = Pipeline(steps=[('RbmModel', RbmModel), ('LogModel', LogModel)])

LogModel.fit(X_train, Y_train)
Classifier.fit(X_train, Y_train)

print ("The RBM model:")
print ("Predict: ", Classifier.predict(X_test))
print ("Real:    ", Y_test)

CM = ConfusionMatrix(Y_test, Classifier.predict(X_test))
CM.print_stats()
