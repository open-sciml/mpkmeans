import numpy as np

def pairwise_q1(X, Y):
    V = (X - Y[:, np.newaxis]) # Y[:, None]
    return np.sum((V * V), axis=2).T



def pairwise_q2(X, Y):
    XX = np.einsum('ij,ij->i', X,  X)
    YY = np.einsum('ij,ij->i', Y,  Y)
    ADD = XX + YY[:, np.newaxis]
    return ADD.T - 2*np.inner(X,  Y)