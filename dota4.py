import dota2api
import requests
import json
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('worked.csv')

train = data[:150]
test = data[150:]
import matplotlib.pyplot as plt
import seaborn as sns

print(len(data))
print(len(test))
print(len(train))

y = np.ravel(train[['outcome']])
X = train.drop('outcome', 1).drop('AHero_0.0', 1).drop('OHero_0.0', 1)

y_test = np.ravel(test[['outcome']])
X_test = test.drop('outcome', 1).drop('AHero_0.0', 1).drop('OHero_0.0', 1)


# clf_svm = svm.SVC(
#     class_weight='balanced'
#     )
# clf_svm.fit(X, y)
# score_svm = cross_val_score(clf_svm, X, y, cv=5).mean()
# print('svm')
# print(score_svm)

# clf_log = LogisticRegression()
# clf_log = clf_log.fit(X,y)
# score_log = cross_val_score(clf_log, X, y, cv=5).mean()
# print('log reg')
# print(score_log)

# clf_pctr = Perceptron(
#     class_weight='balanced'
#     )
# clf_pctr = clf_pctr.fit(X,y)
# score_pctr = cross_val_score(clf_pctr, X, y, cv=5).mean()
# print('perc')
# print(score_pctr)

# clf_knn = KNeighborsClassifier(
#     n_neighbors=10,
#     weights='distance'
#     )
# clf_knn = clf_knn.fit(X,y)
# score_knn = cross_val_score(clf_knn, X, y, cv=5).mean()
# print('knn')
# print(score_knn)

# bagging = BaggingClassifier(
#     KNeighborsClassifier(
#         n_neighbors=2,
#         weights='distance'
#         ),
#     oob_score=True,
#     max_samples=0.5,
#     max_features=1.0
#     )
# clf_bag = bagging.fit(X,y)
# score_bag = clf_bag.oob_score_
# print('bagging?')
# print(score_bag)

# clf_tree = tree.DecisionTreeClassifier(
#     #max_depth=3,\
#     class_weight="balanced",\
#     min_weight_fraction_leaf=0.01\
#     )
# clf_tree = clf_tree.fit(X,y)
# score_tree = cross_val_score(clf_tree, X, y, cv=5).mean()
# print('dec tree')
# print(score_tree)

# clf_rf = RandomForestClassifier(
#     n_estimators=1000, \
#     max_depth=None, \
#     min_samples_split=10 \
#     #class_weight="balanced", \
#     #min_weight_fraction_leaf=0.02 \
#     )
# clf_rf = clf_rf.fit(X,y)
# score_rf = cross_val_score(clf_rf, X, y, cv=5).mean()
# print('rf')
# print(score_rf)

clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
    #class_weight="balanced",
    #min_weight_fraction_leaf=0.02
    )
clf_ext = clf_ext.fit(X,y)
print('extra trees')
score_ext = cross_val_score(clf_ext, X, y, cv=5).mean()
print(score_ext)

print('test test')
score_ext_test = clf_ext.score(X_test,y_test)
print(score_ext_test)

import warnings
warnings.filterwarnings("ignore")

# clf_gb = GradientBoostingClassifier(
#             #loss='exponential',
#             n_estimators=1000,
#             learning_rate=0.1,
#             max_depth=3,
#             subsample=0.5,
#             random_state=0).fit(X, y)
# clf_gb.fit(X,y)
# score_gb = cross_val_score(clf_gb, X, y, cv=5).mean()
# print('gb')
# print(score_gb)

# clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
# clf_ada.fit(X,y)
# score_ada = cross_val_score(clf_ada, X, y, cv=5).mean()
# print('adaboost')
# print(score_ada)

clf = clf_ext
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(scores),np.std(scores)))



# clf_ext = ExtraTreesClassifier(max_features='auto',bootstrap=True,oob_score=True)
# param_grid = { "criterion" : ["gini", "entropy"],
#               "min_samples_leaf" : [1, 5, 10],
#               "min_samples_split" : [8, 10, 12],
#               "n_estimators": [20, 50, 100]}
# gs = GridSearchCV(estimator=clf_ext, param_grid=param_grid, scoring='accuracy', cv=3)
# gs = gs.fit(X,y)
# print(gs.best_score_)
# print(gs.best_params_)

print('--------optimized------------')

clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    criterion='entropy',
    min_samples_leaf=1,
    min_samples_split=8,
    n_estimators=50
    )
clf_ext = clf_ext.fit(X,y)
score_ext = clf_ext.score(X,y)
print(score_ext)
print(pd.DataFrame(list(zip(X.columns, np.transpose(clf_ext.feature_importances_))) \
            ).sort_values(1, ascending=False).head(10))


print('----------------correlations')
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(train, 50))