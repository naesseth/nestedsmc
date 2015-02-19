#!/usr/bin/python
import numpy as np
import helpfunctions as hlp
import nestedSMCinner as nested
import math

class nestedSMC:
    """Class whose constructor generates a particle system to be used
    for nested SMC procedure on the column space.
    
    Constructor
    
    Parameters
    ----------
    t : int
        Year.
    N : int
        Number of particles in column filter. 
    M : int
        Number of particles in row filter. (innermost)
    xCond : 2-D array_like
        Conditional trajectory.
        
    Returns
    -------
    q : object
        Generates a nested SMC object that contains logZ estimate and
        the particle system.
    """
    def __init__(self, t, N, M, xCond=None):
		C1 = 0.5
		C2 = 3.
		
		# Model init
		xDomain = np.arange(2)
				
		# Load parameters
		region = 'dustBowl'
		#region = 'sahel'
		filename = 'parameters/'+region+'Sigma2_N35-55_W90-120_downsampled.csv'
		sigma2 = np.loadtxt(filename, delimiter=',')
		filename = 'parameters/'+region+'MuAb_N35-55_W90-120_downsampled.csv'
		muAb = np.loadtxt(filename, delimiter=',')
		filename = 'parameters/'+region+'MuNorm_N35-55_W90-120_downsampled.csv'
		muNorm = np.loadtxt(filename, delimiter=',')
		filename = 'processedData/'+region+'Yt'+str(t)+'_N35-55_W90-120_downsampled.csv'
		
		Y = np.loadtxt(filename, delimiter=',')
		I = Y.shape[0]
		J = Y.shape[1]
		
		# SMC init
		X = np.zeros( (N, I, J), dtype=bool )
		ancestors = np.zeros( N )
		logZ = 0.
		logW = np.zeros( N )
		w = np.zeros( N )
		ESS = np.zeros( J )
		
		# ---------------
		#      SMC
		# ---------------        
		# SMC first iteration
		params = hlp.params(I = I, muAb = muAb[:,0], muNorm = muNorm[:,0], sigma2 = sigma2[:,0])
		if xCond is not None:
			q = [nested.inner(params, Y[:,0], M, xCond[:,0]) for i in range(N)]
		else:
			q = [nested.inner(params, Y[:,0], M) for i in range(N)]
		logW = np.array([q[i].logZ for i in range(N)])
		maxLogW = np.max(logW)
		w = np.exp(logW - maxLogW)
		logZ = maxLogW + np.log(np.sum(w)) - np.log(N)
		w /= np.sum(w)
		#print w.shape
		ESS[0] = 1/np.sum(w**2)
		#print 'ESS: ',ESS[0]
		#print 'First logZ: ',logZ
		
		ancestors = hlp.resampling(w)
		for i in range(N):
			X[i,:,0] = q[ancestors[i]].simulate()
		
		## SMC MAIN LOOP
		for j in np.arange(1,J):
			#print j
			params = hlp.params(I = I, muAb = muAb[:,j], muNorm = muNorm[:,j], sigma2 = sigma2[:,j])
			if xCond is not None:
				q = [nested.inner(params, Y[:,j], M, xCond[:,j], X[i,:,j-1]) for i in range(N)]
			else:
				q = [nested.inner(params, Y[:,j], M, xSpaceCond = X[i,:,j-1]) for i in range(N)]
			logW = np.array([q[i].logZ for i in range(N)])
			maxLogW = np.max(logW)
			w = np.exp(logW - maxLogW)
			logZ += maxLogW + np.log(np.sum(w)) - np.log(N)
			#print 'j: ',j,' logZ',logZ
			#print 'logW: ',logW
			#print 'Y: ',Y[:,j]
			w /= np.sum(w)
			#print 'Max w: ',np.max(w)
			ESS[j] = 1/np.sum(w**2)
			#print 'ESS: ',ESS[j]
			
			ancestors = hlp.resampling(w)
			for i in range(N):
				X[i,:,j] = q[ancestors[i]].simulate()
		
		#print 'Last logZ: ',logZ
		## Save init to class object
		self.N = N
		self.J = J
		self.I = I
		self.X = X
		self.logZ = logZ
		self.w = w
		self.xCond = xCond
		self.ESS = ESS
        
    def simulate(self, BS=True):
        """Simulate properly weighted sample.
        
        Parameters
        ----------
        BS : bool
            Sample using backward simulation.
            
        Returns
        -------
        Xout : 2-D array_like
            Simulated trajectory.
        """
        if BS:
			C1 = 0.5
			def logPsi(xp,x): return np.sum(C1*(x==xp).astype(float))
			
			w = np.zeros( self.N )
			logW = np.zeros( self.N )
			Xout = np.zeros( (self.I, self.J), dtype=bool )
			
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),1)
			Xout[:,-1] = self.X[b,:,-1]
			for i in np.arange(self.J-1)[::-1]:
				logW = logPsi(Xout[:,i+1],self.X[:,:,i])
				maxLogW = np.max(logW)
				w = np.exp(logW - maxLogW)
				b = hlp.discreteSampling(w, np.arange(self.N), 1)
				Xout[:,i] = self.X[b,:,i]
			
			return Xout
        else:
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),M)
			return self.X[b,:,:]
