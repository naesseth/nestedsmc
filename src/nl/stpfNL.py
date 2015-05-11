import helpfunctions as hlp
import numpy as np
from optparse import OptionParser
import cyResampling as res

def logPhi(x,y): return -0.5*(degFree+1.)*np.log(1+(x-y)**2/degFree)

parser = OptionParser()
parser.add_option("-d", type=int, help="State dimension")
parser.add_option("-N", type=int, help="Particles")
parser.add_option("-M", type=int, help="Nested particles")
parser.add_option("--nrRuns", type=int, help="Number of independent runs")
(args, options) = parser.parse_args()

# Model init
d=args.d
degFree = 10.

filename = './d'+str(d)+'y.csv'
y = np.loadtxt(filename,delimiter=',')

#lindstens dat-file
T = y.shape[1]

#T = y.shape[0]
y = y.T
y = y.reshape((T,d,d))

for j in range(args.nrRuns):
    # Algorithm init
    N = args.N
    M = args.M
    	
    Xcur = np.zeros( (N, M, d, d) )
    Xprev = np.zeros( (N, M, d, d) )
    logW = np.zeros(M)
    w = np.ones( M )
    ancestors = np.zeros( M )

    ESS = np.ones( T )
    outerW = np.ones(N)
    outerAncestors = np.ones(N)
    
    filename = './d'+str(d)+'_N'+str(N)+'_M'+str(M)+'_STPFrun'+str(j+31)+'.csv'
    f = open(filename, 'w')
    f.close()
    
    for t in range(T):
        logZ = np.zeros(N)
        for i in range(N):
            for col in range(d):
                for row in range(d):
                    # Propagate
                    wInd = np.zeros(5)
                    wInd[0] = 1.
                    if row>0:
                        wInd[2] = 0.5
                    if row<d-1:
                        wInd[4] = 0.5
                    if col>0:
                        wInd[1] = 0.5
                    if col<d-1:
                        wInd[3] = 0.5
                    wInd /= np.sum(wInd)
                    for j in range(M):
                        ind = hlp.discreteSampling(wInd, range(5), 1)
                        xMean = np.zeros(M)
                        if ind == 0:
                            xMean[j] = Xprev[i,j,row,col]
                        elif ind == 1:
                            xMean[j] = Xprev[i,j,row,col-1]
                        elif ind == 2:
                            xMean[j] = Xprev[i,j,row-1,col]
                        elif ind == 3:
                            xMean[j] = Xprev[i,j,row,col+1]
                        elif ind == 4:
                            xMean[j] = Xprev[i,j,row+1,col]
                    Xcur[i,:,row,col] = xMean + np.random.normal(size=M)
                    
                    # Weight
                    logW = logPhi(Xcur[i,:,row,col],y[t,row,col])
                    maxLogW = np.max(logW)
                    w = np.exp(logW - maxLogW)
                    logZ[i] += maxLogW + np.log(np.sum(w)) - np.log(M)
                    w /= np.sum(w)
                    
                    # Resample
                    ancestors = res.resampling(w, scheme='systematic')
                    Xprev[i,:,:,:] = Xprev[i,ancestors,:,:]
                    Xcur[i,:,:row,col] = Xcur[i,ancestors,:row,col]
                    Xcur[i,:,:,:col] = Xcur[i,ancestors,:,:col]
                    Xcur[i,:,row,col] = Xcur[i,ancestors,row,col]
        
        # Outer resampling
        maxLogZ = np.max(logZ)
        outerW = np.exp(logZ - maxLogZ)
        outerW /= np.sum(outerW)
        outerAncestors = res.resampling(outerW, scheme='systematic')
        Xprev = Xcur[outerAncestors,:,:,:]
        
        ESS[t] = 1/np.sum(outerW**2)
        f = open(filename, 'a')
        tmpVec = np.r_[t+1, ESS[t], np.mean(np.mean(Xprev,axis=0),axis=0).reshape(d**2), np.mean(np.mean(Xprev**2,axis=0),axis=0).reshape(d**2)]
        np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
        f.close()
