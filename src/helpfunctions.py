#!/usr/bin/python
import numpy as np

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
    return domain[np.digitize(np.random.random_sample(nrSamples), bins)]

def resampling(w, scheme='mult'):
    r"""
    Resampling of particle indices, assume M=N.
    
    Parameters
    ----------
    w : 1-D array_like
        Normalized weights
    scheme : string
        Resampling scheme to use:\n
        mult - Multinomial resampling\n
        res - Residual resampling\n
        strat - Stratified resampling\n
        sys - Systematic resampling
    
    Returns
    -------
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
