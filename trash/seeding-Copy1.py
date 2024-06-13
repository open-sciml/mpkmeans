from scipy.spatial import distance
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh
from distance import *


def d2_seeding1(
    X, num, sample_weight=None, random_state=42, n_trials=None
):
    """seeding of kmeans++ with distance 1

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data.

    num : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each instance of `X`.

    random_state : int
        The random state used to initialize the centers.

    n_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The assignment labels of data points to the closest initial centers. 
        
    corelist : ndarray of shape (num, 2)
        The first column denotes index location of the chosen centers in the data array `X`. For a
        given index and center, X[index] = center. The second column denotes the number of data points 
        allocated to the centers. 
        
    """
    
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=float)
        
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    centers = np.empty((num, n_features), dtype=X.dtype)

    if n_trials is None:
        n_trials = 2 + int(np.log(num))

    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(num, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    
    # pairw_distm = distance.squareform(distance.pdist(X, metric=metric))
    pairw_distm = pairwise_dist1(X)
    
    closest_dist_sq = pairw_distm[center_id, :]
    current_pot = closest_dist_sq @ sample_weight
    
    for c in range(1, num):
       
        rand_vals = random_state.uniform(size=n_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq, dtype=np.float64), rand_vals
        )
        
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = pairw_distm[candidate_ids, :]

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

   
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    
    return X[indices]


def d2_seeding2(
    X, num, sample_weight=None, random_state=42, n_trials=None
):
    """seeding of kmeans++ with distance 2

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data.

    num : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each instance of `X`.

    random_state : int
        The random state used to initialize the centers.

    n_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The assignment labels of data points to the closest initial centers. 
        
    corelist : ndarray of shape (num, 2)
        The first column denotes index location of the chosen centers in the data array `X`. For a
        given index and center, X[index] = center. The second column denotes the number of data points 
        allocated to the centers. 
        
    """
    
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=float)
        
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    centers = np.empty((num, n_features), dtype=X.dtype)

    if n_trials is None:
        n_trials = 2 + int(np.log(num))

    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(num, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    
    # pairw_distm = distance.squareform(distance.pdist(X, metric=metric))
    pairw_distm = pairwise_dist2(X)
    
    closest_dist_sq = pairw_distm[center_id, :]
    current_pot = closest_dist_sq @ sample_weight
    
    for c in range(1, num):
       
        rand_vals = random_state.uniform(size=n_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq, dtype=np.float64), rand_vals
        )
        
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = pairw_distm[candidate_ids, :]

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

   
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    
    return X[indices]



def all_low_d2_seeding1(
    X, num, low_prec, sample_weight=None, random_state=42, n_trials=None
):
    """seeding of kmeans++ with distance 1 in fully low precision

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data.

    num : int
        The number of seeds to choose.
    
    low_prec : chop
        The low precision simulator 
        
    sample_weight : ndarray of shape (n_samples,)
        The weights for each instance of `X`.

    random_state : int
        The random state used to initialize the centers.

    n_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The assignment labels of data points to the closest initial centers. 
        
    corelist : ndarray of shape (num, 2)
        The first column denotes index location of the chosen centers in the data array `X`. For a
        given index and center, X[index] = center. The second column denotes the number of data points 
        allocated to the centers. 
        
    """
    
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=float)
        
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    centers = np.empty((num, n_features), dtype=X.dtype)

    if n_trials is None:
        n_trials = 2 + int(np.log(num))

    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(num, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    
    # pairw_distm = distance.squareform(distance.pdist(X, metric=metric))
    pairw_distm = all_low_pairwise_dist1(X, low_prec)
    
    closest_dist_sq = pairw_distm[center_id, :]
    current_pot = closest_dist_sq @ sample_weight
    
    for c in range(1, num):
       
        rand_vals = random_state.uniform(size=n_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq, dtype=np.float64), rand_vals
        )
        
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = pairw_distm[candidate_ids, :]

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

   
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    
    return X[indices]


def all_low_d2_seeding2(
    X, num, low_prec, sample_weight=None, random_state=42, n_trials=None
):
    """seeding of kmeans++ with distance 2 in fully low precision

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data.

    num : int
        The number of seeds to choose.

    low_prec : chop
        The low precision simulator 
        
    sample_weight : ndarray of shape (n_samples,)
        The weights for each instance of `X`.

    random_state : int
        The random state used to initialize the centers.

    n_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The assignment labels of data points to the closest initial centers. 
        
    corelist : ndarray of shape (num, 2)
        The first column denotes index location of the chosen centers in the data array `X`. For a
        given index and center, X[index] = center. The second column denotes the number of data points 
        allocated to the centers. 
        
    """
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=float)
        
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    centers = np.empty((num, n_features), dtype=X.dtype)

    if n_trials is None:
        n_trials = 2 + int(np.log(num))

    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(num, -1, dtype=int)
    centers[0] = low_prec(X[center_id])
    indices[0] = center_id
    
    # pairw_distm = distance.squareform(distance.pdist(X, metric=metric))
    pairw_distm = all_low_pairwise_dist2(X, low_prec)
    
    closest_dist_sq = pairw_distm[center_id, :]
    current_pot = closest_dist_sq @ sample_weight
    
    for c in range(1, num):
       
        rand_vals = random_state.uniform(size=n_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq, dtype=np.float64), rand_vals
        )
        
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = pairw_distm[candidate_ids, :]

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

   
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = low_prec(X[best_candidate])
        indices[c] = best_candidate
    
    return X[indices]



def pca_aggregate1(data, tol=0.5): 
    """aggregate the data with distance 1

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).

    sorting : str
        The sorting method for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.

    tol : float
        The tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    """
    splist = list() # store the starting points
    len_ind = data.shape[0]
    fdim = data.shape[1]

    # change to svd 
    if fdim > 1:
        if fdim <= 3: # memory inefficient
            gemm = get_blas_funcs("gemm", [data.T, data])
            _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
            sort_vals = data@U1.reshape(-1)
        else:
            U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]

    else:
        sort_vals = data[:,0]

    sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output

    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind]

    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        else:
            clustc = data[i,:] 
            labels[i] = lab
            num_group = 1

        for j in range(i+1, len_ind):
            if labels[j] >= 0:
                continue

            if (sort_vals[j] - sort_vals[i] > tol):
                break       

            dat = clustc - data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  

        lab += 1
        
    centers = list()
    for i in np.unique(labels):
        centers.append(np.mean(data[labels == i], axis=0))
        
    centers = np.array(centers)
    return centers


def pca_aggregate2(data, tol=0.5):
    """aggregate the data with distance 2

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    tol : float
        The tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  
    
    
    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    half_nrm2 (numpy.ndarray):
        Precomputed values for distance computation.

    """
    
    len_ind, fdim = data.shape

    # get sorting values
    if fdim>1:
        if fdim <= 3: # memory inefficient
            gemm = get_blas_funcs("gemm", [data.T, data])
            _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
            sort_vals = data@U1.reshape(-1)
        else:
            U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
    else:
        sort_vals = data[:,0]

    sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind] 
 
    half_r2 = 0.5*tol**2
    half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5 # precomputation

    lab = 0
    labels = [-1] * len_ind
    nr_dist = 0 
    splist = list()

    for i in range(len_ind): 
        if labels[i] >= 0:
            continue

        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1

        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
        last_j = np.searchsorted(sort_vals, tol + sort_vals[i], side='right')
        ips = np.matmul(data[i+1:last_j,:], clustc.T)
        
        for j in range(i+1, last_j):
            if labels[j] >= 0:
                continue

            nr_dist += 1
            if half_nrm2[j] - ips[j-i-1] <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))
        lab += 1

    centers = list()
    for i in np.unique(labels):
        centers.append(np.mean(data[labels == i], axis=0))
        
    centers = np.array(centers)
    return centers



def all_low_pca_aggregate1(data, tol, low_prec):
    """aggregate data with distance 1 in fully low precision

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    tol : float
        The tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  
    
    low_prec : chop
        The low precision simulator 
        
    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    half_nrm2 (numpy.ndarray):
        Precomputed values for distance computation.

    """
    data = low_prec(data)
    len_ind, fdim = data.shape

    # get sorting values
    if fdim>1:
        if fdim <= 3: # memory inefficient
            gemm = get_blas_funcs("gemm", [data.T, data])
            _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
            sort_vals = data@U1.reshape(-1)
        else:
            U1, s1, _ = svds(data.astype(float), k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
    else:
        sort_vals = data[:,0]

    sort_vals = low_prec(sort_vals*np.sign(-sort_vals[0])) # flip to enforce deterministic output
    
    ind = np.argsort(sort_vals)
    data = low_prec(data[ind])
    sort_vals = sort_vals[ind] 
 
    half_r2 = 0.5*tol**2
    half_nrm2 = low_prec(np.einsum('ij,ij->i', data, data) * 0.5) # precomputation

    lab = 0
    labels = [-1] * len_ind
    nr_dist = 0 
    splist = list()

    for i in range(len_ind): 
        if labels[i] >= 0:
            continue

        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1

        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
        last_j = np.searchsorted(sort_vals, tol + sort_vals[i], side='right')
        ips = low_prec(np.matmul(data[i+1:last_j,:], clustc.T))
        
        for j in range(i+1, last_j):
            if labels[j] >= 0:
                continue

            nr_dist += 1
            if half_nrm2[j] - ips[j-i-1] <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))
        lab += 1

    centers = list()
    for i in np.unique(labels):
        centers.append(np.mean(data[labels == i], axis=0))
    centers = np.array(centers)
    
    return low_prec(centers)


def all_low_pca_aggregate2(data, tol, low_prec): 
    """aggregate data with distance 2 in fully low precision

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).

    sorting : str
        The sorting method for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.

    tol : float
        The tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    low_prec : chop
        The low precision simulator 
        
    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    fdim = data.shape[1]
    
    data = low_prec(data)
    
    # change to svd 
    if fdim > 1:
        if fdim <= 3: # memory inefficient
            gemm = get_blas_funcs("gemm", [data.T, data])
            _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
            sort_vals = data@U1.reshape(-1)
        else:
            U1, s1, _ = svds(data.astype(float), k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]

    else:
        sort_vals = data[:,0]

    sort_vals = low_prec(sort_vals*np.sign(-sort_vals[0])) # flip to enforce deterministic output

    ind = np.argsort(sort_vals)
    data = low_prec(data[ind])
    sort_vals = sort_vals[ind]

    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        else:
            clustc = data[i,:] 
            labels[i] = lab
            num_group = 1

        for j in range(i+1, len_ind):
            if labels[j] >= 0:
                continue

            if (sort_vals[j] - sort_vals[i] > tol):
                break       

            dat = low_prec(clustc - data[j,:])
            dist = low_prec(np.inner(dat, dat))
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  

        lab += 1

    centers = list()
    for i in np.unique(labels):
        centers.append(np.mean(data[labels == i], axis=0))
    centers = np.array(centers)
    
    return low_prec(centers)