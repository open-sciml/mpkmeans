import numpy as np

def pairwise_dist_y1(x, y):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :] = standard_distance1(y, x[i])
    return distm


def pairwise_dist_y2(x, y):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :] = standard_distance2(y, x[i])
    return distm


def mp1_pairwise_dist_y1(x, y, low_prec):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :], low_op = mp_distance1(y, x[i], low_prec)
    return distm


def mp2_pairwise_dist_y2(x, y, low_prec):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :], low_op = mp_distance2(y, x[i], low_prec)
    return distm


def all_low_pairwise_dist_y1(x, y, low_prec):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :] = all_low_distance1(y, x[i], low_prec)
    return distm


def all_low_pairwise_dist_y2(x, y, low_prec):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :] = all_low_distance2(y, x[i], low_prec)
    return distm


def pairwise_dist1(x):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:] = standard_distance1(x[i+1:, :], x[i])
        distm[i+1:, i] = distm[i, i+1:]
    return distm

def pairwise_dist2(x):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:] = standard_distance2(x[i+1:, :], x[i])
        distm[i+1:, i] = distm[i, i+1:]
    return distm

def mp1_pairwise_dist(x, low_prec):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:], low_op = mp_distance1(x[i+1:, :], x[i], low_prec)
        distm[i+1:, i] = distm[i, i+1:]
    return distm


def all_low_pairwise_dist1(x, low_prec):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:] = all_low_distance1(x[i+1:, :], x[i], low_prec)
        distm[i+1:, i] = distm[i, i+1:]
    return distm


def mp2_pairwise_dist(x, low_prec):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:], low_op = mp_distance2(x[i+1:, :], x[i], low_prec)
        distm[i+1:, i] = distm[i, i+1:]
    return distm


def all_low_pairwise_dist2(x, low_prec):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i+1:] = all_low_distance2(x[i+1:, :], x[i], low_prec)
        distm[i+1:, i] = distm[i, i+1:]
    return distm


def mp_distance1(x, y, low_prec, delta=10e-3):
    v = x - y
    low_op = np.linalg.norm(v, ord=1, axis=1) > delta
    dist = np.zeros(x.shape[0])
    v_p = low_prec(v[low_op])
    v = v[~low_op]
    dist[low_op] = low_prec(np.sum(v_p * v_p, axis=1))
    dist[~low_op] = np.sum(v * v, axis=1)
    return dist, low_op


def all_low_distance1(x, y, low_prec):
    x, y = low_prec(x), low_prec(y)
    v = low_prec(x - y)
    dist = np.sum(v * v, axis=1)
    return dist


def mp_distance2(x, y, low_prec, delta=10e-3):
    xx = np.einsum('ij,ij->i', x,  x)
    
    cct = np.inner(y, y)
    low_op = np.abs(xx - cct) > delta**2
    
    dist = np.zeros(x.shape[0])
    dist[low_op] = 2*low_prec(np.inner(low_prec(x[low_op]), low_prec(y)))
    
    dist[~low_op] = 2*np.inner(x[~low_op], y)
    dist = xx + cct - dist
    return dist, low_op

    
def all_low_distance2(x, y, low_prec):
    x = low_prec(x)
    y = low_prec(y)
    xx = np.einsum('ij,ij->i', x,  x)
    
    cct = low_prec(np.inner(y, y))
    dist = np.zeros(x.shape[0])
    dist = low_prec(2*np.inner(x, y))
    dist = low_prec(low_prec(xx + cct) - dist)
    return dist


def standard_distance1(x, y):
    v = x - y 
    return np.sum(v * v, axis=1)


def standard_distance2(x, y):
    xx = np.einsum('ij,ij->i', x,  x)
    
    cct = np.inner(y, y)
    dist = np.zeros(x.shape[0])
    dist = 2*np.inner(x, y)
    dist = xx + cct - dist
    return dist
