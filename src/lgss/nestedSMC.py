#!/usr/bin/python
import numpy as np
import helpfunctions as hlp

class nestedFAPF:
    r"""Class to generate properly weighted samples from high-dimensional
    linear Gaussian state space model.
    
    Constructor
    
    Parameters
    ----------
    params : model
        Struct-like (see helpfunctions) containing model parameters.
    """
    def __init__(self, params, Y, N, xCond=None):
		# Model init
		d = len(Y)
		def logPhi(x,y): return -0.5*params.tauPhi*(x-y)**2
		
		# SMC init
		if xCond is None:
			xCond = np.zeros( d )
		logZ = 0.
		X = np.zeros( (N, d) )
		Xa = np.zeros( (N, d) )
		ancestors = np.zeros( N )
		w = np.zeros( (N, d) )
		W = np.zeros( (N, d) )
		logW = np.zeros( N )
		#ESS = np.zeros( d )
		#NT = N/2
		#resamp = 0
		
		# i=1
		X[:,0] = params.tauRho*params.a*xCond[0] + (1/np.sqrt(params.tauRho))*np.random.normal(size=N)
		
		# Weighting
		logW = logPhi(X[:,0],Y[0])
		maxLogW = np.max(logW)
		w[:,0] = np.exp(logW - maxLogW)
		logZ += maxLogW + np.log(np.sum(w[:,0])) - np.log(N)
		w[:,0] /= np.sum(w[:,0])
		ancestors = hlp.resampling(w[:,0])
		X[:,0] = X[ancestors,0]
		Xa[:,0] = X[:,0] 
		#tempW = w[:,0] / np.sum(w[:,0])
		#ESS[0] = 1/np.sum(tempW**2)
		
		#if ESS[0] < NT:
			#logZ += maxLogW + np.log(np.sum(w[:,0])) - np.log(N)
			
			#w[:,0] /= np.sum(w[:,0])
			#ancestors = hlp.resampling(w[:,0],scheme='sys')
			#X[:,0] = X[ancestors,0] 
			
			#logW = np.zeros( N )
		#else:
			#ancestors = np.arange(N)
				
		# i=2:d
		for i in np.arange(1,d):
			# Propagate
			tau = params.tauRho + params.tauPsi
			mu = (params.tauRho*params.a*xCond[i] + params.tauPsi*X[:,i-1])/tau
			X[:,i] = mu + (1/np.sqrt(tau))*np.random.normal(size=N)
			
			# Weighting, Resampling
			logW = logPhi(X[:,i],Y[i])
			maxLogW = np.max(logW)
			w[:,i] = np.exp(logW - maxLogW)
			logZ += maxLogW + np.log(np.sum(w[:,i])) - np.log(N)
			w[:,i] /= np.sum(w[:,i])
			ancestors = hlp.resampling(w[:,i])
			X[:,i] = X[ancestors,i] 
			Xa[:,:i] = Xa[ancestors,:i]
			Xa[:,i] = X[:,i]
			#tempW = w[:,i] / np.sum(w[:,i])
			#ESS[i] = 1/np.sum(tempW**2)
			
			## ESS-based Resampling
			#if ESS[i] < NT or i == d-1:
				#logZ += maxLogW + np.log(np.sum(w[:,i])) - np.log(N)
				
				#w[:,i] /= np.sum(w[:,i])
				#ancestors = hlp.resampling(w[:,i],scheme='sys')
				#X[:,i] = X[ancestors,i] 
				
				#logW = np.zeros( N )
			#else:
				#ancestors = np.arange(N)
		
		# Save init to class object
		self.N = N
		self.d = d
		self.X = X
		self.Xa = Xa
		self.Y = Y
		self.params = params
		self.logZ = logZ
		self.w = w
		self.xCond = xCond
		#self.nrResamp = np.sum(ESS < NT)
        
    def simulate(self, M, BS=True):
        r"""Simulate properly weighted sample.
        
        Parameters
        ----------
        BS : bool
            Sample using backward simulation.
            
        Returns
        -------
        Xout : 1-D array_like
            Simulated trajectory.
        """
        if BS:
			def logPsi(xp,x): return -0.5*self.params.tauPsi*(xp-x)**2
			w = np.zeros( self.N )
			logW = np.zeros( self.N )
			Xout = np.zeros( (M, self.d) )
			
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),M)
			Xout[:,-1] = self.X[b,-1]
			for i in np.arange(self.d-1)[::-1]:
				for j in np.arange(M):
					logW = logPsi(Xout[j,i+1],self.X[:,i])
					maxLogW = np.max(logW)
					w = np.exp(logW - maxLogW)
					b = hlp.discreteSampling(w, np.arange(self.N), 1)
					Xout[j,i] = self.X[b,i]
			
			return Xout
        else:
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),M)
			return self.Xa[b,:]
			
			#def logPhi(x,y): return -0.5*self.params.tauPhi*(x-y)**2
			#def logRho(xp,x): return -0.5*self.params.tauRho*(xp-self.params.a*x)**2
			#def logPsi(xp,x): return -0.5*self.params.tauPsi*(xp-x)**2
			#w = np.zeros( self.N )
			#logW = np.zeros( self.N )
			
			#Xout = np.zeros( (M, self.d) )
			## Draw B_n uniformly on 1:N
			##b = hlp.discreteSampling(np.ones(self.N),range(self.N),M)
			#b = hlp.discreteSampling(self.w,range(self.N),M)
			
			#Xout[:,-1] = self.X[b,-1]
			#for i in np.arange(self.d-1)[::-1]:
				#for j in np.arange(M):
					#logW = logPhi(self.X[:,i],self.Y[i]) + logRho(self.X[:,i],self.xCond[i]) + logPsi(Xout[j,i+1],self.X[:,i])
					#maxLogW = np.max(logW)
					#w = np.exp(logW - maxLogW)
					#b = hlp.discreteSampling(w, range(self.N), 1)
					#Xout[j,i] = self.X[b,i]
			
			#return Xout
		
