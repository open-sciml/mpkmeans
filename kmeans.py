import numpy as np
from sklearn.cluster._k_means_common import _inertia_dense
from distance import *
from seeding import *

class StandardKMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers, labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            print(_iter)
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = pairwise_dist_y(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers

    
class KMeansplusplus:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.centers = d2_seeding(X, self.n_clusters)
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers, labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            print(_iter)
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = pairwise_dist_y(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers
    
    
    
class mp1KMeansplusplus:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.centers = mp1_d2_seeding(X, self.n_clusters)
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers, labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            print(_iter)
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = mp1_pairwise_dist_y(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers
    
    
    

class mp2KMeansplusplus:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.centers = mp2_d2_seeding(X, self.n_clusters)
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers, labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            print(_iter)
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = mp2_pairwise_dist_y(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers