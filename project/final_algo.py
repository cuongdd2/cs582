import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

init = 100 # start after 10 rounds avoid bias
end = init + 380 * 5 - 100
data = pd.read_csv('./data/data.csv')
data.sort_values(by=['season'])
data = data[init:end]
data = data[
    ['home_team_name', 'away_team_name', 'home_team_goal', 'away_team_goal', 'home_team_shoton', 'away_team_shoton',
     'home_team_shotoff', 'away_team_shotoff']]
data['diff'] = data.apply(lambda row: np.sign(row['home_team_goal'] - row['away_team_goal']), axis=1)

# add home win/draw/lost prob
for i in range(init, end):
    home = data[0:i].query("home_team_name=='%s'" % data.loc[i, 'home_team_name'])
    away = data[0:i].query("away_team_name=='%s'" % data.loc[i, 'away_team_name'])
    home_match = home.shape[0]
    away_match = away.shape[0]

    if home_match > 0:
        data.loc[i, 'hw'] = home.query("diff==1").shape[0] / home_match
        data.loc[i, 'hd'] = home.query("diff==0").shape[0] / home_match
        data.loc[i, 'hl'] = home.query("diff==-1").shape[0] / home_match
        data.loc[i, 'hon'] = home[0:i]['home_team_shoton'].sum() / home_match
        data.loc[i, 'hoff'] = home[0:i]['home_team_shotoff'].sum() / home_match
        # data.loc[i, 'hp'] = home[0:i]['home_possession'].sum() / home_match
    else:
        data.loc[i, 'hw'] = data.loc[i, 'hd'] = data.loc[i, 'hl'] = data.loc[i, 'hon'] = data.loc[i, 'hoff'] = 0

    if away_match > 0:
        data.loc[i, 'aw'] = away.query("diff==1").shape[0] / away_match
        data.loc[i, 'ad'] = away.query("diff==0").shape[0] / away_match
        data.loc[i, 'al'] = away.query("diff==-1").shape[0] / away_match
        data.loc[i, 'aon'] = home[0:i]['away_team_shoton'].sum() / away_match
        data.loc[i, 'aoff'] = home[0:i]['away_team_shotoff'].sum() / away_match
        # data.loc[i, 'ap'] = home[0:i]['away_possession'].sum() / away_match
    else:
        data.loc[i, 'aw'] = data.loc[i, 'ad'] = data.loc[i, 'al'] = data.loc[i, 'aon'] = data.loc[i, 'aoff'] = 0

cols = ['hw', 'hd', 'hl', 'aw', 'ad', 'al', 'hon', 'hoff', 'aon', 'aoff']
# cols = ['hw', 'hd', 'hl', 'aw', 'ad', 'al']
features = data.loc[:, cols]
# features.to_csv("./data/processed_data.csv")
features = np.float32(features)
labels = data['diff']

# training dataset
train_features = features[100:int(end * 0.8)]
train_labels = labels[100:int(end * 0.8)]

# test dataset
test_features = features[-int(end * 0.2):]
test_labels = labels[-int(end * 0.2):]

# test models
classifiers = [
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(20, min_samples_split=5),
    GaussianProcessClassifier(),
    MLPClassifier(),
    GaussianNB(),
    AdaBoostClassifier()
]

'''
SVC:    0.521604938272
kNN:    0.515432098765
DT :    0.614197530864
RF :    0.682098765432  0.688271604938  0.657407407407
GP :    0.530864197531
MLP:    0.543209876543
GNB:    0.54012345679
Ada:    0.537037037037
'''

for clf in classifiers:
    clf.fit(train_features, train_labels)
    print(clf.score(test_features, test_labels))



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
