import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

A = np.array([[1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5]])
B = np.array([[1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]])
X = np.append(A.T, B.T, axis=1)
K = 2

def kmeans_display(X, label):
    K = np.amax(label) + 1
    colors = ['b^', 'go', 'rs']
    for i in range(K):
        X0 = X[label == i, :]
        # plt.plot(X0[:, 0], X0[:, 1], markersize=4, alpha=.8)
        plt.plot(X0[:, 0], X0[:, 1], colors[i], markersize=4, alpha=.8)

    plt.axis('equal')
    # plt.plot()
    plt.show()

# random init centroids
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


# calculate distance
# return label based on function to get x that make y min
def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)

# init centers with zero size K x X dimension
# calculate mean by labels
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    return np.sum(np.sum(centers - new_centers)) != 0
    # return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])


def kmeans(X, K):
    # centers = [kmeans_init_centers(X, K)]
    centers = np.array([[1.0, 1.0], [5.0, 7.0]])
    it = 0
    while True and it < 100:
        labels = kmeans_assign_labels(X, centers)
        new_centers = kmeans_update_centers(X, labels, K)
        if has_converged(centers, new_centers):
            break
        centers = new_centers
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print(centers)
kmeans_display(X, labels)
