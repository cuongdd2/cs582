import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('./data/data.csv')
home = data.get('home_team_goal')
away = data.get('away_team_goal')
index = data.groupby(['home_team_goal', 'away_team_goal']).size().index
x = index.map(lambda t: t[0])
y = index.map(lambda t: t[1])
areas = data.groupby(['home_team_goal', 'away_team_goal']).size()
plt.scatter(x, y, s=areas * 4, alpha=1)
plt.xlabel('home')
plt.ylabel('away')
plt.show()

# cm = plt.cm.get_cmap('jet')
#
# fig, ax = plt.subplots()
# sc = ax.scatter(x, y, s=z * 500, c=z, cmap=cm, linewidth=0, alpha=0.5)
# ax.grid()
# fig.colorbar(sc)
# plt.show()
#
# plt.plot(X, y, '.')
