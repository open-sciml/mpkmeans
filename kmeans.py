import numpy as np
from sklearn.cluster._k_means_common import _inertia_dense
from distance import *
from seeding import *
from sklearn.preprocessing import normalize

"""
StandardKMeans1: the native kmeans algorithm using distance 1
StandardKMeans2: the native kmeans algorithm using distance 2
mp1KMeans: the mixed precision kmeans algorithm using distance 1
mp2KMeans: the mixed precision kmeans algorithm using distance 2
allowKMeans1: kmeans performed in full low precision using distance 1
allowKMeans2: kmeans performed in full low precision using distance 2
"""



class StandardKMeans1:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = d2_seeding1(X, self.n_clusters)
        else:
            self.centers = pca_aggregate1(normalize(X, norm="l1"), self.alpha)
            
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
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = pairwise_dist_y1(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.centers.shape[0])])
        return new_centers

    
class StandardKMeans2:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = d2_seeding2(X, self.n_clusters)
        else:
            self.centers = pca_aggregate2(normalize(X, norm="l1"), self.alpha)
            
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
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = pairwise_dist_y2(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.centers.shape[0])])
        return new_centers

    
class mp1KMeans:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, low_prec=None, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        self.low_prec = low_prec
        
        
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = all_low_d2_seeding1(X, self.n_clusters, self.low_prec)
        else:
            self.centers = all_low_pca_aggregate1(normalize(X, norm="l1"), self.alpha, self.low_prec)
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            print(updated_centers.shape)
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = mp1_pairwise_dist_y1(X, self.centers, self.low_prec)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([self.low_prec(self.low_prec(X[labels == i]).mean(axis=0)) for i in range(self.centers.shape[0])])
        return new_centers

    
    
    
class mp2KMeans:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, low_prec=None, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        self.low_prec = low_prec
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = all_low_d2_seeding2(X, self.n_clusters, self.low_prec)
        else:
            self.centers = all_low_pca_aggregate2(normalize(X, norm="l1"), self.alpha, self.low_prec)
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = mp2_pairwise_dist_y2(X, self.centers, self.low_prec)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([self.low_prec(self.low_prec(X[labels == i]).mean(axis=0)) for i in range(self.centers.shape[0])])
        return new_centers
    
    
    
    
class allowKMeans1:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, low_prec=None, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        self.low_prec = low_prec
        
        
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = all_low_d2_seeding1(X, self.n_clusters, self.low_prec)
        else:
            self.centers = all_low_pca_aggregate1(normalize(X, norm="l1"), self.alpha, self.low_prec)
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = all_low_pairwise_dist_y1(X, self.centers, self.low_prec)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([self.low_prec(self.low_prec(X[labels == i]).mean(axis=0)) for i in range(self.centers.shape[0])])
        return new_centers

    
    
    
class allowKMeans2:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', alpha=0.5, tol=1e-4, low_prec=None, verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.verbose = verbose
        self.random_state = random_state
        self.low_prec = low_prec
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = all_low_d2_seeding2(X, self.n_clusters, self.low_prec) 
        else:
            self.centers = all_low_pca_aggregate2(normalize(X, norm="l1"), self.alpha, self.low_prec)
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(X, simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=1)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = all_low_pairwise_dist_y2(X, self.centers, self.low_prec)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([self.low_prec(self.low_prec(X[labels == i]).mean(axis=0)) for i in range(self.centers.shape[0])])
        return new_centers
    
    
    
    

class chop(object):
    def __init__(self, xtype):
        self.xtype = xtype
        
    def __call__(self, x):
        return self.xtype(x)
