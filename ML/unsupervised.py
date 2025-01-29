import numpy as np
from ML.costs import distortion
from utils.utils import multivariate_gaussian

class KMeans:

    def __init__(self, K):
        self.K = K


    def findClosestCentroid(self, X, centroids):
        '''
        X         --> (m, n)     
        centroids --> (K, n)

        return:
        closest_centroids --> (m, 1)
        '''
        m, n = X.shape
        closest_centroids = np.zeros((m, 1))

        #loop through all examples
        for i in range(m):
            distance = np.zeros(self.K)

            distance = np.sum(np.square(X[i] - centroids), axis=1)
            best_centroid = np.argmin(distance)
            closest_centroids[i] = best_centroid

        return closest_centroids
    
    
    def calculateNewCentroids(self, X, closest_centroids):
        '''
        X --> (m, n)
        closest_centroids --> (m, 1)

        return:
        new_centroids --> (K, n)
        '''

        m, n = X.shape
        new_centroids = np.zeros((self.K, n))

        for i in range(self.K):
            grouped_points = X[closest_centroids.reshape(X.shape[0]) == i, :] #all the points part of the ith cluster
            grouped_points_mean = grouped_points.mean(axis=0)
            new_centroids[i] = grouped_points_mean

        return new_centroids

    

    def initializeRandomCentroids(self, X):
        '''
        X --> (m, n)

        return:
        centroids --> (K, n)
        '''
        m, n = X.shape
        indx_arr = np.arange(0, m)
        np.random.shuffle(indx_arr)
        centroids = (X[indx_arr])[:self.K]

        return centroids
    
    
    def fit(self, X, max_iters=10, max_tries=10, verbose=False):
        '''
        X --> (m, n)
        
        return:
        best_init_centroids --> (K, n)   best centroid positions
        best_closest_centroids --> (m, ) points grouped according to best centroids
        '''
        m, n = X.shape

        costs = np.zeros(max_tries)
        closest_centroids_arr = np.zeros((max_tries, m), dtype=int)
        init_centroids_arr = np.zeros((max_tries, self.K, n))

        for try_ in range(max_tries):
            init_centroids = self.initializeRandomCentroids(X)

            for i in range(max_iters):
                closest_centroids = self.findClosestCentroid(X, init_centroids)
                init_centroids = self.calculateNewCentroids(X, closest_centroids)

            closest_centroids_arr[try_] = closest_centroids.reshape(m)
            init_centroids_arr[try_] = init_centroids
            costs[try_] = distortion(X, closest_centroids, init_centroids, self.K)

            if verbose == True:
                print(f"Tries: {try_+1}/{max_tries}")

        # choose the best configuration 
        best = np.argmin(costs)
        best_closest_centroids = (closest_centroids_arr[best])
        best_init_centroids = init_centroids_arr[best]

        return best_init_centroids, best_closest_centroids
        


class AnomalyDetection:

    def __init__(self, mean=None, std=None, p_x=None):
        self.mean = mean
        self.std = std
        self.p_x = p_x

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.p_x = multivariate_gaussian(X, self.mean, self.std)

    def predict(self, X, epsilon=1e-7):
        m, n = X.shape
        y = np.zeros(m)

        y = self.p_x < epsilon
        
        return y

