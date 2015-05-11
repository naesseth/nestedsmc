#!/usr/bin/python
"""

Class nested SMC.

"""
import numpy as np
import helpfunctions as hlp
import nestedSMCinner as nested
import math
import cyResampling as res

def forwardFiltering(xPrev, xCond, N):
    
    return messages

class nestedSMC:
    def __init__(self, t, N, xCond=None):
        def phi(x,y,sig2,mu_ab,mu_norm): return np.exp( -0.5*(y-mu_ab*x.astype('float')-mu_norm*(1.-x.astype('float')))**2/sig2 )
        def rho(xp,x): return np.exp( C2*(xp.astype('bool') == x.astype('bool')).astype('float') )
        def psi(xp,x): return np.exp( C1*(xp == x).astype('float') )
        
        C1 = 0.5
        C2 = 3.
                
        # Model init
        xDomain = np.arange(2)
        psiMat = np.array([np.exp( C1*(xDomain.astype('bool') == False).astype('float') ),np.exp( C1*(xDomain.astype('bool') == True).astype('float') )])
                                
        # Load parameters
        #region = 'dustBowl'
        region = 'sahel'
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
        msg = np.zeros( (N,I,2) )
        c = np.zeros( (N,I) )
        # ---------------
        #      SMC
        # ---------------
               
        # SMC first iteration, j = 0
        
        # Forward filtering
        unaryFactor = np.ones( (N, I, 2) )
        for n in range(N):
            unaryFactor[n,0,:] *= phi( xDomain, Y[0,0], sigma2[0,0], muAb[0,0], muNorm[0,0] )
            unaryFactor[n,0,:] *= rho( xDomain, xCond[0,0] )
            msg[n,0,:] = np.dot(psiMat, unaryFactor[n,0,:])
            c[n,0] = np.sum(msg[n,0,:])
            msg[n,0,:] /= c[n,0]
        
            for i in np.arange(1,I-1):
                unaryFactor[n,i,:] *= phi( xDomain, Y[i,0], sigma2[i,0], muAb[i,0], muNorm[i,0] )
                unaryFactor[n,i,:] *= rho( xDomain, xCond[i,0] )
                msg[n,i,:] = np.dot(psiMat, unaryFactor[n,i,:]*msg[n,i-1,:])
                c[n,i] = np.sum(msg[n,i,:])
                msg[n,i,:] /= c[n,i]
            unaryFactor[n,I-1,:] *= phi( xDomain, Y[I-1,0], sigma2[I-1,0], muAb[I-1,0], muNorm[I-1,0] )
            unaryFactor[n,I-1,:] *= rho( xDomain, xCond[I-1,0] )
            
        # Backward sampling
        for n in range(N):
            tempDist = unaryFactor[n,I-1,:]*msg[n,I-2,:]
            tempDist /= np.sum(tempDist)
            X[n,I-1,0] = hlp.discreteSampling(tempDist, xDomain, 1)
            
            for i in np.arange(2,I-1)[::-1]:
                tempDist = unaryFactor[n,i,:]*msg[n,i-1,:]
                tempDist /= np.sum(tempDist)
                X[n,i,0] = hlp.discreteSampling(tempDist*psiMat[:,X[n,i+1,0]], xDomain, 1)
                
        logW = np.sum(np.log(c[:,:I-1]),axis=1) + np.log(np.sum(unaryFactor[:,I-1,:]*msg[:,I-2,:],axis=1))
        maxLogW = np.max(logW)
        w = np.exp(logW - maxLogW)
        logZ += maxLogW + np.log(np.sum(w)) - np.log(N)
        
        
        # SMC iteration j = 1 to J
        for j in np.arange(1,J):
            # Forward filtering
            unaryFactor = np.ones( (N, I, 2) )
            for n in range(N):
                unaryFactor[n,0,:] *= phi( xDomain, Y[0,j], sigma2[0,j], muAb[0,j], muNorm[0,j] )
                unaryFactor[n,0,:] *= rho( xDomain, xCond[0,j] )
                unaryFactor[n,0,:] *= psi( xDomain, X[n,0,j-1] )
                msg[n,0,:] = np.dot(psiMat, unaryFactor[n,0,:])
                c[n,0] = np.sum(msg[n,0,:])
                msg[n,0,:] /= c[n,0]
            
                for i in np.arange(1,I-1):
                    unaryFactor[n,i,:] *= phi( xDomain, Y[i,j], sigma2[i,j], muAb[i,j], muNorm[i,j] )
                    unaryFactor[n,i,:] *= rho( xDomain, xCond[i,j] )
                    unaryFactor[n,i,:] *= psi( xDomain, X[n,i,j-1] )
                    msg[n,i,:] = np.dot(psiMat, unaryFactor[n,i,:]*msg[n,i-1,:])
                    c[n,i] = np.sum(msg[n,i,:])
                    msg[n,i,:] /= c[n,i]
                
                unaryFactor[n,I-1,:] *= phi( xDomain, Y[I-1,j], sigma2[I-1,j], muAb[I-1,j], muNorm[I-1,j] )
                unaryFactor[n,I-1,:] *= rho( xDomain, xCond[I-1,j] )
                unaryFactor[n,I-1,:] *= psi( xDomain, X[n,I-1,j-1] )
                
            logW = np.sum(np.log(c[:,:I-1]),axis=1) + np.log(np.sum(unaryFactor[:,I-1,:]*msg[:,I-2,:],axis=1))
            maxLogW = np.max(logW)
            w = np.exp(logW - maxLogW)
            logZ += maxLogW + np.log(np.sum(w)) - np.log(N)
            ancestors = res.resampling(w,'stratified')
            
            # Backward sampling
            for n in range(N):
                tempDist = unaryFactor[ancestors[n],I-1,:]*msg[ancestors[n],I-2,:]
                tempDist /= np.sum(tempDist)
                X[n,I-1,j] = hlp.discreteSampling(tempDist, xDomain, 1)
                
                for i in np.arange(2,I-1)[::-1]:
                    tempDist = unaryFactor[ancestors[n],i,:]*msg[ancestors[n],i-1,:]
                    tempDist /= np.sum(tempDist)
                    X[n,i,j] = hlp.discreteSampling(tempDist*psiMat[:,X[n,i+1,j]], xDomain, 1)
                    
            
        
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
        if BS:
            def logPsi(xp,x): return np.sum(C1*(xp==x).astype(float),axis=1)
            C1 = 0.5
                        
            w = np.zeros( self.N )
            logW = np.zeros( self.N )
            Xout = np.zeros( (self.I, self.J), dtype=bool )
                        
            b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),1)
            Xout[:,-1] = self.X[b,:,-1]
            for j in np.arange(self.J-1)[::-1]:
                logW = logPsi(Xout[:,j+1],self.X[:,:,j])
                maxLogW = np.max(logW)
                w = np.exp(logW - maxLogW)
                b = hlp.discreteSampling(w, np.arange(self.N), 1)
                Xout[:,j] = self.X[b,:,j]
                        
            return Xout
        else:
            b = hlp.discreteSampling(np.ones(self.N),np.arange(self.N),M)
            return self.X[b,:,:]
