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


def pairwise_mix_prec(X, Y, low_prec, delta=DELTA):
    XX = np.einsum('ij,ij->i', X,  X)
    YY = np.einsum('ij,ij->i', Y,  Y)
    
    divisor = XX[:, np.newaxis] / YY
    
    condition = (divisor > delta) | (1/divisor > delta)
    
    ADD = XX + YY[:, np.newaxis]
    dotXY = np.inner(X,  Y)
    dotXY[condition] = low_prec(dotXY[condition])
    return ADD.T - 2*dotXY, np.sum(condition)/(X.shape[0]*Y.shape[0])


def pairwise_low_prec(X, Y):
    XX = low_prec(np.einsum('ij,ij->i', X,  X))
    YY = low_prec(np.einsum('ij,ij->i', Y,  Y))
    ADD = low_prec(XX + YY[:, np.newaxis])
    return low_prec(ADD.T - 2*np.inner(X,  Y))
