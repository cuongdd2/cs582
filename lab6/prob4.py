import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def has_converged(centers, new_centers):
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])


def kmeans(X, K):
    # centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    centroids = np.array([[1.0, 1.0], [5.0, 7.0]])
    it = 0
    while True:
        it += 1
        D = cdist(X, centroids)
        labels = np.argmin(D, axis=1)
        new_centroids = np.zeros((K, X.shape[1]))
        for k in range(K):
            new_centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        display(X, K, labels)
        plt.show()
        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def display(X, K, labels):
    for i in range(K):
        X0 = X[labels == i, :]
        plt.plot(X0[:, 0], X0[:, 1], '.')


def error(X, K, labels):
    sum = 0
    for i in range(K):
        X0 = X[labels == i, :]
        sum += np.std(X0)
    print(sum / K)


def random_data():
    for i in range(6):
        mean = 200 * np.random.random_sample((1, 2))
        X0 = np.random.multivariate_normal(mean[0], [[10, 0], [0, 10]], np.random.randint(20, 50))
        if i == 0:
            X = X0
        else:
            X = np.concatenate((X, X0))
    return X

from sklearn.cluster import KMeans

A = np.array([[1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5]])
B = np.array([[1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]])
X = np.append(A.T, B.T, axis=1)
# X = random_data()
for K in range(2, 10):
    (labels, centroids) = kmeans(X, K)
    display(X, K, labels)
    plt.show()
    error(X, K, labels)
    cls = KMeans(n_clusters=K, random_state=0)
    cls.fit(X)
    lbl = cls.labels_
    display(X, K, lbl)
    plt.show()
