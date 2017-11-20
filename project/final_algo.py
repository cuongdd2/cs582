import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = pd.read_csv('./data/data.csv')
data.sort_values(by=['season'])
init = 380 * 4
end = init + 380
data = data[init:end]
data = data[['home_team_name', 'away_team_name', 'home_team_goal', 'away_team_goal']]
data['diff'] = data.apply(lambda row: np.sign(row['home_team_goal'] - row['away_team_goal']), axis=1)

# add home win/draw/lost prob
# after 20 matches (2 rounds) we have enough data
for i in range(init, end):
    home = data[0:i].query("home_team_name=='%s'" % data.loc[i, 'home_team_name'])
    away = data[0:i].query("away_team_name=='%s'" % data.loc[i, 'away_team_name'])
    home_match = home.shape[0]
    away_match = away.shape[0]

    if home_match > 0:
        data.loc[i, 'hw'] = home.query("diff==1").shape[0] / home_match
        data.loc[i, 'hd'] = home.query("diff==0").shape[0] / home_match
        data.loc[i, 'hl'] = home.query("diff==-1").shape[0] / home_match
    else:
        data.loc[i, 'hw'] = data.loc[i, 'hd'] = data.loc[i, 'hl'] = 0

    if away_match > 0:
        data.loc[i, 'aw'] = away.query("diff==1").shape[0] / away_match
        data.loc[i, 'ad'] = away.query("diff==0").shape[0] / away_match
        data.loc[i, 'al'] = away.query("diff==-1").shape[0] / away_match
    else:
        data.loc[i, 'aw'] = data.loc[i, 'ad'] = data.loc[i, 'al'] = 0

# data.to_csv("./data/processed_data.csv")
labels = data['diff']
# features = data.loc[:,['hw', 'hd', 'hl', 'aw', 'ad', 'al']]
features = data.loc[:,['hw',  'hl', 'aw', 'al']]
train_features = features[80:340]
train_labels = labels[80:340]
test_features = features[-40:]
test_labels = labels[-40:]

# test models
names = ["SVC: ", "GaussianProcess:", "MLP:", "GaussianNB:"]
classifiers = [
    SVC(),
    GaussianProcessClassifier(),
    MLPClassifier(),
    GaussianNB(),
]
i = 0
for clf in classifiers:
    clf.fit(train_features, train_labels)
    print(names[i], clf.score(test_features, test_labels))
    i += 1
# test = data[-50:]
# training = data[:330]
# training.mean()

# create percent w, d, l of when home
# create percent w, d, l of when away
# training = data[:-10]
# print(data.mean())
# data = data.groupby("home_team_name")
# print(data.mean())
# training = data[:190]
# test = data[10:]
# for i in range(0, 380, 10):
#     print(data[:i].mean())

