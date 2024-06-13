import numpy as np
from sklearn.cluster._k_means_common import _inertia_dense
from .dist import *
from .seeding import *

"""
==========================================================================================
    The packges of k-means algorithms
-----------------------------------------------------------------------------------------
    StandardKMeans1: the native kmeans algorithm using distance (5.2)
    StandardKMeans2: the native kmeans algorithm using distance (5.3)
    mpKMeans: the mixed precision kmeans algorithm based on distance computing of Algorithm (5.1)
    allowKMeans1: the mixed kmeans performed with distance (5.2) in full low precision
    allowKMeans2: the mixed kmeans performed with distance (5.3) in full low precision 
==========================================================================================
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
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            self.centers = pca_aggregate1((X - xmin)/(xmax - xmin) , self.alpha)
            self.centers = self.centers*(xmax - xmin) + xmin
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        labels = self.labels_assignment(X)
        self.inertia.append(_inertia_dense(np.double(X), 
                                      simulate_weights,
                                      self.centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10
                                     )
                           )
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            self.iter = _iter
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(np.double(X), simulate_weights, 
                                      updated_centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10)
            
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        # dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dists = pairwise_q1(X, self.centers)
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
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            self.centers = pca_aggregate2((X - xmin)/(xmax - xmin), self.alpha)
            self.centers = self.centers*(xmax - xmin) + xmin
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        labels = self.labels_assignment(X)
        self.inertia.append(_inertia_dense(np.double(X), 
                                      simulate_weights,
                                      self.centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10
                                     )
                           )
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            # if np.all(self.centers == new_centers):
            #    break
            
            self.iter = _iter
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(np.double(X), simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=10)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        dists = pairwise_q2(X, self.centers)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.centers.shape[0])])
        return new_centers


    

class mpKMeans:
    def __init__(self, n_clusters=2, max_iters=100, seeding='d2', 
                 alpha=0.5, tol=1e-4, low_prec=None, delta=1.5, 
                 verbose=0, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.inertia = list()
        self.labels = None
        self.alpha = alpha
        self.seeding = seeding
        self.delta = delta
        self.verbose = verbose
        self.random_state = random_state
        self.low_prec = low_prec
        
        
        
    def fit(self, X):
        np.random.seed(self.random_state)
        if self.seeding == 'random':
            self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.seeding == 'd2':
            self.centers = mp_low_d2_seeding2(X, self.n_clusters, self.low_prec)
        else:
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            self.centers = all_low_pca_aggregate1((X - xmin)/(xmax - xmin), self.alpha, self.low_prec)
            self.centers = self.centers*(xmax - xmin) + xmin
        
        low_prec_trigger = list()
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        labels, lct = self.labels_assignment(X)
        low_prec_trigger.append(np.mean(lct))
        self.inertia.append(_inertia_dense(np.double(X), 
                                      simulate_weights,
                                      self.centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10
                                     )
                           )
        for _iter in range(self.max_iters):
            labels, lct = self.labels_assignment(X)
            low_prec_trigger.append(np.mean(lct))
            updated_centers = self.centers_update(X, labels)
            # if np.all(self.centers == new_centers):
            #    break
            if np.sum(np.isnan(updated_centers)) > 1:
                break
            
            self.iter = _iter
            
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(np.double(X), simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=10)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels, _ = self.labels_assignment(X)
        self.low_prec_trigger = np.mean(low_prec_trigger)
        
    def labels_assignment(self, X):
        dists, low_prec_trigger = pairwise_mix_prec(X, self.centers, self.low_prec, self.delta)
        return np.argmin(dists, axis=1), low_prec_trigger
    
    def centers_update(self, X, labels):
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.centers.shape[0])])
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
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            self.centers = all_low_pca_aggregate1((X - xmin)/(xmax - xmin), self.alpha, self.low_prec)
            self.centers = self.centers*(xmax - xmin) + xmin
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        labels = self.labels_assignment(X)
        self.inertia.append(_inertia_dense(np.double(X), 
                                      simulate_weights,
                                      self.centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10
                                     )
                           )
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            if np.sum(np.isnan(updated_centers)) > 1:
                break
            # if np.all(self.centers == new_centers):
            #    break
            
            self.iter = _iter
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(np.double(X), simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=10)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        dists = pairwise_low_prec_q1(X, self.centers, self.low_prec)
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
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            self.centers = all_low_pca_aggregate2((X - xmin)/(xmax - xmin), self.alpha, self.low_prec)
            self.centers = self.centers*(xmax - xmin) + xmin
            
        simulate_weights = np.ones(X.shape[0], dtype=np.float64)
        labels = self.labels_assignment(X)
        self.inertia.append(_inertia_dense(np.double(X), 
                                      simulate_weights,
                                      self.centers.astype(np.float64), 
                                      labels.astype(np.int32), n_threads=10
                                     )
                           )
        for _iter in range(self.max_iters):
            labels = self.labels_assignment(X)
            updated_centers = self.centers_update(X, labels)
            
            if np.sum(np.isnan(updated_centers)) > 1:
                break
            # if np.all(self.centers == new_centers):
            #    break
            
            self.iter = _iter
            if ((self.centers - updated_centers)**2).sum() <= self.tol:
                break
                
            _inertia = _inertia_dense(np.double(X), simulate_weights, updated_centers.astype(np.float64), labels.astype(np.int32), n_threads=10)
            self.inertia.append(_inertia)
            self.centers = updated_centers
            
            if self.verbose:
                print("iteration:", _iter+1)
                
        if self.verbose:
            print("total clusters:", self.centers.shape[0])
            
        self.labels = self.labels_assignment(X)
        
        
    def labels_assignment(self, X):
        dists = pairwise_low_prec_q2(X, self.centers, self.low_prec)
        return np.argmin(dists, axis=1)
    
    def centers_update(self, X, labels):
        new_centers = np.array([self.low_prec(self.low_prec(X[labels == i]).mean(axis=0)) for i in range(self.centers.shape[0])])
        return new_centers
    
    
    
    

class chop(object):
    def __init__(self, xtype):
        self.xtype = xtype
        
    def __call__(self, x):
        return self.xtype(x)