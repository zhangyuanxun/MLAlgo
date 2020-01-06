from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import copy

def compute_distance(p, centers):
    k = centers.shape[0]
    min_dis = float('inf')
    cluster = None

    for i in range(k):
        d = (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2
        if min_dis > d:
            min_dis = d
            cluster = i

    return cluster


def KMean(X, k):
    m = X.shape[0]
    n = X.shape[1]

    # random init
    centers = np.zeros((k, n))
    max_x = np.min(X[:, 0])
    min_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    for i in range(k):
        centers[i, 0] = np.random.uniform(min_x, max_x)
        centers[i, 1] = np.random.uniform(min_y, max_y)

    init_centers = copy.deepcopy(centers)

    # define clusters
    for i in range(100):
        clusters_cnt = [0] * k
        new_centers = np.zeros((k, 2))
        for p in X:
            cluster = compute_distance(p, centers)
            new_centers[cluster] += p
            clusters_cnt[cluster] += 1

        for cluster in range(k):
            new_centers[cluster] = new_centers[cluster] / clusters_cnt[cluster]

        centers = new_centers

    return init_centers, centers


n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

plt.figure(figsize=(12, 12))

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[:, 0], X[:, 1])
plt.title("Number of Blobs")

init_centers, centers = KMean(X, k=3)
plt.scatter(init_centers[:, 0], init_centers[:, 1], marker='^', c='red', s=1000)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=1000)
plt.show()