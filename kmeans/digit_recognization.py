import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmeans.display_network import display_network
from scipy.misc import imshow


data = MNIST("./data/")
data.load_testing()
X = data.test_images
K = 10

kmeans = KMeans(n_clusters=K).fit(X)

pred_modal = kmeans.predict(X)
A = display_network(kmeans.cluster_centers_.T, K, 1)
plt.imshow(A, interpolation='nearest', cmap='jet')
plt.show()

cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

img = cmap(norm(A))
imshow(img)