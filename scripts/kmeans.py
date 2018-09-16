import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)


def plot_random_data():
    '''Plot random data generated from different multivariate distributions'''
    colors = [
        'orange', 'red', 'cyan', 'green', 'blue', 'magenta',
        'lightgreen', 'yellow', 'lightyellow', 'lightblue', 'white'
    ]
    means = np.array([[0, 0],
                      [3, 3],
                      [-2, -2],
                      [0, 4],
                      [3, 0],
                      [-2, 2],
                      [1, 2],
                      [2, -2],
                      [-4, 0],
                      [-4, 4]])
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    for i in range(10):
        X = np.random.multivariate_normal(mean=means[i],
                                          cov=[[0.5, 0], [0, 0.5]],
                                          size=150)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[i])
    plt.xlim([-9, 8])
    ax.set_aspect('equal')
    ax.grid('false')
    plt.show()
