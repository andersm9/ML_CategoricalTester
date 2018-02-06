#https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, average_precision_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.externals import joblib
#import data to Pandas Frame:
dataset_url = 'iris.data'
data = pd.read_csv(dataset_url, sep = ',')
##data looks like this:
#"s_length","s_width","p_length","p_width","classifier"
#5.1,3.5,1.4,0.2,Iris-setosa
#4.9,3.0,1.4,0.2,Iris-setosa
#4.7,3.2,1.3,0.2,Iris-setosa


#seperate training and target data - "classifier" is taken from the data (above):
y = data.classifier
X = data.drop('classifier', axis=1)



#create list for models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RFC', RandomForestClassifier(n_estimators=100)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('MLP', MLPClassifier(max_iter=1200)))
models.append(('ADB', AdaBoostClassifier()))
models.append(('GPC', GaussianProcessClassifier()))



for name, model in models:
    # label encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    # binary encode
    ##OneHotEncoding of target
    #onehot_encoder = OneHotEncoder(sparse=False)
    #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)  
    #create training and test data:
    X_train, X_test, y_train, y_test = train_test_split(X, integer_encoded, test_size=0.3, random_state=123, stratify=y)

    print(name)
    #create pipeline - normalize data and select method
    pipeline = make_pipeline(preprocessing.StandardScaler(), model)
    #display available hyperparameters
    #print(pipeline.get_params())
    #define hyperparamaters to tune
    #hyperparameters1 = {'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
    #              'randomforestclassifier__max_depth': [None, 10, 5, 3, 1] }
    hyperparameters = {}
    #carry out cross validation pipeline (tests training data against all hyperparameter permutations)
    clf = GridSearchCV(pipeline, hyperparameters, cv=25)
    # Fit and tune model
    clf.fit(X_train, y_train)

    #predict target against test data
    y_pred = clf.predict(X_test)
    #test prediction against actual test data
    #APS is for classification
    print("average_precision_score")
    print(accuracy_score(y_test, y_pred))
    #print(average_precision_score(y_test, y_pred))
    #print(f1_score(y_test, y_pred))


