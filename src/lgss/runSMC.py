import helpfunctions as hlp
import numpy as np
import scipy.stats as stat
import scipy.special as spec
import nestedFAPF2 as nsmc
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", type=int, help="State dimension")
parser.add_option("--tauPhi", type=float, help="Measurement precision")
parser.add_option("-N", type=int, help="Particles")
(args, options) = parser.parse_args()

def logPhi(x,y): return -0.5*tauPhi*(x-y)**2
tauPhi = args.tauPhi
d=args.d

filename = 'simulatedData/d'+str(d)+'tauPhi'+str(tauPhi)+'y.txt'
y = np.loadtxt(filename)
T = y.shape[1]
a = 0.5
tauPsi = 1.
tauRho = 1.

N = args.N
NT = N/2
Xcur = np.zeros( (N, d) )
Xprev = np.zeros( (N, d) )
logW = np.zeros(N)
w = np.ones( N )
ancestors = np.zeros( N )
ESS = N*np.ones( T*d )

filename = './results/d'+str(d)+'_N'+str(N)+'tauPhi'+str(tauPhi)+'_nipsSMC.csv'
f = open(filename, 'w')
f.close()


for t in range(T):
    if ESS[t*d-1] < NT:
        ancestors = hlp.resampling(w,scheme='sys')
        Xprev = Xprev[ancestors,:]
        w = np.ones( N )
        
    Xcur[:, 0] = a*Xprev[:, 0] + (1/np.sqrt(tauRho))*np.random.normal(size=N)
    logW = logPhi(Xcur[:,0],y[0,t])
    maxLogW = np.max(logW)
    w *= np.exp(logW - maxLogW)
    w /= np.sum(w)
    
    ESS[t*d] = 1/np.sum(w**2)
    
    for i in np.arange(1,d):
        # Resampling
        if ESS[t*d+i-1] < NT:
            ancestors = hlp.resampling(w,scheme='sys')
            Xprev = Xprev[ancestors,:]
            Xcur[:,:i] = Xcur[ancestors,:i]
            w = np.ones( N )
            
        # Propagate
        tau = tauRho + tauPsi
        mu = (tauRho*a*Xprev[:, i] + tauPsi*Xcur[:,i-1])/tau
        Xcur[:,i] = mu + (1/np.sqrt(tau))*np.random.normal(size=N)
        
        # Weighting
        logW = logPhi(Xcur[:,i],y[i,t])
        maxLogW = np.max(logW)
        w *= np.exp(logW - maxLogW)
        w /= np.sum(w)
        
        ESS[t*d+i] = 1/np.sum(w**2)
    
    Xprev = Xcur
    f = open(filename, 'a')
    tmpVec = np.r_[t+1, np.sum(np.tile(w,(d,1)).T*Xcur,axis=0), np.sum(np.tile(w,(d,1)).T*Xcur**2,axis=0)]
    np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
    f.close()
