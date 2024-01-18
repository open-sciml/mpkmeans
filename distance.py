import numpy as np

"""
working precision u, float32
ul: precision lower than u
"""
def pairwise_dist_y(x, y):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :] = standard_distance(y, x[i])
    return distm


def mp1_pairwise_dist_y(x, y):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :], low_op = mp_distance1(y, x[i])
    return distm


def mp2_pairwise_dist_y(x, y):
    distm = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        distm[i, :], low_op = mp_distance2(y, x[i])
    return distm



def pairwise_dist(x):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i:] = standard_distance(x[i:, :], x[i])
        distm[i:, i] = distm[i, i:]
    return distm

def mp1_pairwise_dist(x):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i:], low_op = mp_distance1(x[i:, :], x[i])
        distm[i:, i] = distm[i, i:]
    return distm

def mp2_pairwise_dist(x):
    distm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        distm[i, i:], low_op = mp_distance2(x[i:, :], x[i])
        distm[i:, i] = distm[i, i:]
    return distm


def mp_distance1(x, y, delta=10e-3):

    v = x.astype(np.float32) - y.astype(np.float32)
    v = v.astype(np.float32)
    low_op = np.linalg.norm(v, ord=1, axis=1) > delta
    dist = np.zeros(x.shape[0])
    dist = np.sum(v * v, axis=1)
    dist[low_op] = np.sum(
        v[low_op].astype(np.float16) * v[low_op].astype(np.float16), 
        axis=1).astype(np.float16)
    dist[~low_op] = dist[~low_op].astype(np.float16)
    return dist.astype(np.float32), low_op


def mp_distance2(x, y, delta=10e-3):
    x = x.astype(np.float32)
    y = y.astype(np.float32) 
    xx = np.einsum('ij,ij->i', x,  x).astype(np.float32)
    
    cct = np.inner(y, y).astype(np.float32)
    low_op = np.abs(xx - cct) > delta**2
    
    dist = np.zeros(x.shape[0])
    dist[low_op] = 2*np.inner(x[low_op].astype(np.float16), 
                              y.astype(np.float16)
                             ).astype(np.float16)
    
    dist[~low_op] = 2*np.inner(x[~low_op], y).astype(np.float32)
    dist = xx + cct - dist
    return dist.astype(np.float32), low_op
    
    
def standard_distance(x, y):
    v = x - y 
    return np.sum(v * v, axis=1)