#!/usr/bin/python
"""

Class nested SMC.

"""
import numpy as np
import helpfunctions as hlp
import math

class inner:
	def __init__(self, params, y, N, xTimeCond=None, xSpaceCond=None):
		C1 = 0.5
		C2 = 3.

		# Model init
		xDomain = np.arange(2)
		def logPhi(x,y,sig2,mu_ab,mu_norm): return -0.5*(y-mu_ab*x.astype('float')-mu_norm*(1.-x.astype('float')))**2/sig2

		I = params.I
		muAb = params.muAb
		muNorm = params.muNorm
		sigma2 = params.sigma2
		
		# SMC init
		X = np.zeros( (N, I), dtype=bool )
		ancestors = np.zeros( N )
		logZ = 0.
		logW = np.zeros( N )
		w = np.zeros( N )
		
		# ---------------
		#      SMC
		# ---------------        
		# Sample proposal
		tempDist = np.zeros(2)
		if xTimeCond is not None:
			tempDist += C2*(xTimeCond[0] == xDomain.astype(bool))
		if xSpaceCond is not None:
			tempDist += C1*(xSpaceCond[0] == xDomain.astype(bool))
		tempDist = np.exp(tempDist)
		tempDist /= np.sum(tempDist)
		X[:,0] = hlp.discreteSampling(tempDist, xDomain, N)
		
		# Weighting
		logW = logPhi(X[:,0], y[0], sigma2[0], muAb[0], muNorm[0])
		maxLogW = np.max(logW)
		w = np.exp(logW-maxLogW)
		logZ = maxLogW + np.log(np.sum(w)) - np.log(N)
		w /= np.sum(w)
		ancestors = hlp.resampling(w)
		X[:,0] = X[ancestors,0]
		
			
		## SMC MAIN LOOP
		for i in np.arange(1,I):
			tempDist = np.zeros(2)
			if xTimeCond is not None:
				tempDist += C2*(xTimeCond[i] == xDomain.astype(bool))
			if xSpaceCond is not None:
				tempDist += C1*(xSpaceCond[i] == xDomain.astype(bool))
			for iParticle in range(N):
				tempParticleDist = tempDist+C1*(X[iParticle,i-1] == xDomain.astype(bool))
				tempParticleDist = np.exp(tempParticleDist)
				tempParticleDist /= np.sum(tempParticleDist)
				X[iParticle,i] = hlp.discreteSampling(tempParticleDist, xDomain, 1)
			logW = logPhi(X[:,i], y[i], sigma2[i], muAb[i], muNorm[i])
			maxLogW = np.max(logW)
			w = np.exp(logW-maxLogW)
			logZ += maxLogW + np.log(np.sum(w)) - np.log(N)
			#if math.isnan(logZ):
				#print 'X: ',X[:,i]
				#print 'y: ',y[i]
				#print 'muAb: ',muAb[i]
				#print 'muNorm: ',muNorm[i]
				#print 'sig2: ',sigma2[i]
				#raw_input()
			w /= np.sum(w)
			ancestors = hlp.resampling(w)
			X[:,i] = X[ancestors,i]
		
		## Save init to class object
		self.N = N
		self.C1 = C1
		self.I = I
		self.X = X
		self.y = y
		self.logZ = logZ
		self.w = w
		
	def simulate(self, BS=True):
		if BS:
			C1 = 0.5
			def logPsi(xp,x): return C1*(x == xp)
			w = np.zeros( self.N )
			logW = np.zeros( self.N )
			Xout = np.zeros( self.I )
			
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),1)
			Xout[-1] = self.X[b,-1]
			for i in np.arange(self.I-1)[::-1]:
				logW = logPsi(Xout[i+1],self.X[:,i])
				maxLogW = np.max(logW)
				w = np.exp(logW - maxLogW)
				b = hlp.discreteSampling(w, np.arange(self.N), 1)
				Xout[i] = self.X[b,i]
			
			return Xout
		else:
			b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),M)
			return self.Xa[b,:]

