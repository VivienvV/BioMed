import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import svm
import numpy as np


def preprocess_data(train_call, train_clin):
    full_data = train_call
    train_arr = train_call[train_call.columns[-100:]].to_numpy()
    train_arr = train_arr.T
    l = train_clin[train_clin.columns[-1:]].to_numpy().T.squeeze()
    labels = np.array(pd.factorize(l)[0].tolist())
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(train_arr, labels)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    cols = model.get_support(indices=True)
    train_arr = model.transform(train_arr.astype(np.float32))
    
    return train_arr, labels, full_data.iloc[cols]