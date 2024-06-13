import numpy as np
DELTA = 1

def pairwise_q1(X, Y): # Eq~(4.2)
    V = (X - Y[:, np.newaxis]) # Y[:, None]
    return np.sum((V * V), axis=2).T


def pairwise_q2(X, Y): # Eq~(4.3)
    XX = np.einsum('ij,ij->i', X,  X)
    YY = np.einsum('ij,ij->i', Y,  Y)
    ADD = XX + YY[:, np.newaxis]
    return ADD.T - 2*np.inner(X,  Y)


def pairwise_low_prec_q1(X, Y, low_prec):
    V = low_prec(low_prec(X) - low_prec(Y[:, np.newaxis])) 
    return low_prec(np.sum(low_prec(V * V), axis=2).T)


def pairwise_low_prec_q2(X, Y, low_prec):
    X = low_prec(X)
    Y = low_prec(Y)
    XX = low_prec(np.einsum('ij,ij->i', X,  X))
    YY = low_prec(np.einsum('ij,ij->i', Y,  Y))
    ADD = low_prec(XX + YY[:, np.newaxis])
    return low_prec(ADD.T - 2*low_prec(np.inner(X,  Y)))


def pairwise_mix_prec(X, Y, low_prec, delta=DELTA):
    XX = np.einsum('ij,ij->i', X,  X)
    YY = np.einsum('ij,ij->i', Y,  Y)

    divisor = XX[:, np.newaxis] / YY

    condition = (divisor >= delta) | (1/divisor >= delta)

    ADD = XX + YY[:, np.newaxis]
    dotXY = np.inner(X,  Y)

    scale1 = 1 / np.max(dotXY,axis=1).reshape(-1,1)
    scale2 = 1 / np.max(dotXY,axis=0).reshape(-1,1)
    dotXY = scale1 * dotXY * scale2.reshape(-1)

    values = dotXY[condition]
    dotXY[condition] = low_prec(values)
    dotXY = (1/scale1.reshape(-1,1)) * dotXY * (1/scale2.reshape(-1)) 
    result = ADD.T - 2*dotXY
    return result, np.sum(condition)/(X.shape[0]*Y.shape[0])



"""
def scaled_dot(X, Y, low_prec):
    scale1 = 1/np.max(X,axis=1).reshape(-1,1)
    scale2 = 1/np.max(Y,axis=1).reshape(-1,1)
    inner_prod = low_prec(np.inner(low_prec(scale1*X),  low_prec(scale2*Y)))
    return ((1/scale1.reshape(-1,1)) * inner_prod) *(1/scale2.reshape(-1,1)) 


def scaled_dot(X, Y, low_prec):
    scale1 = 1/np.max(X)
    scale2 = 1/np.max(Y)
    inner_prod = np.inner(scale1*X,  scale2*Y)
    return (inner_prod/scale1) /scale2

def scaled_dot(X, Y, low_prec):
    scale1 = 1/np.linalg.norm(X, ord=2, axis=1).reshape(-1,1)
    scale2 = 1/np.linalg.norm(Y, ord=2, axis=1).reshape(-1,1)
    inner_prod = np.inner(scale1*X,  scale2*Y)
    return ((1/scale1.reshape(-1,1)) * inner_prod) *(1/scale2.reshape(-1,1)) 
"""
