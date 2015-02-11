#!/usr/bin/python
"""

Helpfunctions for SMC methods.

"""
import numpy as np
from numpy.random import random_sample

class params:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)
		
class model:
	def __init__(self, mdl, dim, N, logTarget, simProposal=None, logProposal=None):
		self.mdl = mdl
		
		self.dim = dim
		self.N = N
		self.logTarget = logTarget
		self.simProposal = simProposal
		self.logProposal = logProposal

def discreteSampling(weights, domain, nrSamples):
    weights /= np.sum(weights)
    bins = np.cumsum(weights)
    return domain[np.digitize(random_sample(nrSamples), bins)]

def resampling(w, scheme='mult'):
    """
    Resampling of particle indices, assume M=N.
    
    Parameters
    ----------
    w : 1-D array_like
    Normalized weights
    scheme : string
    Resampling scheme to use {mult, res, strat, sys}:
    mult - Multinomial resampling
    res - Residual resampling
    strat - Stratified resampling
    sys - Systematic resampling
    
    Output
    ------
    ind : 1-D array_like
    Indices of resampled particles.
    """
     
    N = w.shape[0]
    ind = np.arange(N)
    
    # Multinomial
    if scheme=='mult':
        ind = discreteSampling(w, np.arange(N), N)
    # Residual
    elif scheme=='res':
        R = np.sum( np.floor(N * w) )
        if R == N:
            ind = np.arange(N)
        else:
            wBar = (N * w - np.floor(N * w)) / (N-R)
            Ni = np.floor(N*w) + np.random.multinomial(N-R, wBar)
            iter = 0
            for i in range(N):
                ind[iter:iter+Ni[i]] = i
                iter += Ni[i]
    # Stratified
    elif scheme=='strat':
        u = (np.arange(N)+np.random.rand(N))/N
        wc = np.cumsum(w)
        ind = np.arange(N)[np.digitize(u, wc)]
    # Systematic
    elif scheme=='sys':
        u = (np.arange(N) + np.random.rand(1))/N
        wc = np.cumsum(w)
        k = 0
        for i in range(N):
            while (wc[k]<u[i]):
                k += 1
            ind[i] = k
    else:
        raise Exception("No such resampling scheme.")
    return ind

def cpf_as(f, h, y, q, r, N, X=None):
    """
    Conditional particle filter for hidden Markov models using ancestor sampling.
    
    Parameters
    ----------
    f : lambda function
    Process equation.
    h : lambda function
    Measurement equation.
    y : 1-D array_like
    Measurements
    q : float
    Process noise variance.
    r : float
    Measurement noise variance.
    N : integer
    Number of particles.
    X : 1-D array_like
    Conditioned particles, if not provided unconditional PF is run.
    
    Output
    ------
    x : 2-D array_like
    Output particles.
    w : 1-D array_like
    Particle weights.
    """    
    
    T = len(y)
    x = np.zeros( (N,T) )
    a = np.zeros( (N,T), dtype=np.int )
    w = np.zeros( (N,T) )
    
    # Initialize particles
    x[:,0] = 0.
    if X is not None:
        x[N-1,0] = X[0]
    
    for t in range(T):
        if t > 0:
            ind = resampling(w[:,t-1])
            xpred = f(x[:,t-1],t-1)
            x[:,t] = xpred[ind] + np.sqrt(q)*np.random.randn(N)
            if X is not None:
                x[N-1,t] = X[t]
                # Ancestor sampling
                m = np.exp(-1/(2*q)*(X[t]-xpred)**2)
                w_as = w[:,t-1] * m
                w_as /= np.sum(w_as)
                ind[N-1] = discreteSampling(w_as, range(N), 1)
            # Store ancestor indices
            a[:,t] = ind
            
        # Compute important weights
        ypred = h(x[:,t])
        logweights = -1/(2*r)*(y[t] - ypred)**2
        const = np.max(logweights)
        weights = np.exp( logweights - const )
        w[:,t] = weights / np.sum(weights)

    # Generate trajectories from ancestor indices
    ind = a[:,T-1]
    for t in range(T-1)[::-1]:
        x[:,t] = x[ind,t]
        ind = a[ind,t]
    
    return x, w

# Slightly faster for high-dim (order of a few %)    
#def discreteSampling2(weights, domain, nrSamples):
#    weights /= np.sum(weights)
#    bins = np.cumsum(weights)
#    return domain[np.searchsorted(bins, random_sample(nrSamples))]

# Slowest
#def multiDimDiscreteSampling(weights, domain, nrSamples):
#    weights = weights / weights.sum(axis=0)[np.newaxis,:]
#    bins = np.cumsum(weights,axis=0)
#    return domain[np.apply_along_axis(lambda s: s.searchsorted(random_sample(nrSamples)), axis=0,arr=bins)]

#def ravel_multi_index(coord, shape):
    #return coord[0] * shape[1] + coord[1]

#def unravel_index(coord, shape):
    #iy = np.remainder(coord, shape[1])
    #ix = (coord - iy) / shape[1]
    #return ix, iy
