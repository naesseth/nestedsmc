import helpfunctions as hlp
import numpy as np
import nestedFAPF2 as nsmc
from optparse import OptionParser

def logPhi(x,y): return -0.5*tauPhi*(x-y)**2

parser = OptionParser()
parser.add_option("-d", type=int, help="State dimension")
parser.add_option("--tauPhi", type=float, help="Measurement precision")
parser.add_option("-N", type=int, help="Particles")
parser.add_option("-M", type=int, help="Nested particles")
parser.add_option("--nrRuns", type=int, help="Number of independent runs")
(args, options) = parser.parse_args()

# Model init
tauPhi = args.tauPhi
d=args.d
a = 0.5
tauPsi = 1.
tauRho = 1.

filename = 'simulatedData/d'+str(d)+'tauPhi'+str(tauPhi)+'y.txt'
y = np.loadtxt(filename)
T = y.shape[1]

for j in range(args.nrRuns):
	# Algorithm init
	N = args.N
	M = args.M
	
	Xcur = np.zeros( (N, M, d) )
	Xprev = np.zeros( (N, M, d) )
	logW = np.zeros(M)
	w = np.ones( M )
	ancestors = np.zeros( M )

	ESS = np.ones( T )
	outerW = np.ones(N)
	outerAncestors = np.ones(N)
	logZ = np.zeros(N)
	
	
	filename = './results/paper/d'+str(d)+'_N'+str(N)+'_M'+str(M)+'tauPhi'+str(tauPhi)+'_STPFrun'+str(j+100+1)+'.csv'
	f = open(filename, 'w')
	f.close()
	
	for t in range(T):
		for i in range(N):
			# j = 1
			# Propagate
			Xcur[i,:,0] = a*Xprev[i,:,0] + (1/np.sqrt(tauRho))*np.random.normal(size=M)
			
			# Weight
			logW = logPhi(Xcur[i,:,0],y[0,t])
			maxLogW = np.max(logW)
			w = np.exp(logW - maxLogW)
			logZ[i] = maxLogW + np.log(np.sum(w))-np.log(M)
			w /= np.sum(w)
			
			# Resample
			ancestors = hlp.resampling(w)
			Xprev[i,:,:] = Xprev[i,ancestors,:]
			Xcur[i,:,0] = Xcur[i,ancestors,0]
			
			# j = 2:d
			for j in np.arange(1,d):
				# Propagate
				tau = tauRho + tauPsi
				mu = (tauRho*a*Xprev[i, :, j] + tauPsi*Xcur[i,:,j-1])/tau
				Xcur[i,:,j] = mu + (1/np.sqrt(tau))*np.random.normal(size=M)
			
				# Weighting
				logW = logPhi(Xcur[i, :, j],y[j,t])
				maxLogW = np.max(logW)
				w = np.exp(logW - maxLogW)
				logZ[i] += maxLogW + np.log(np.sum(w))-np.log(M)
				w /= np.sum(w)
				
				# Resampling
				ancestors = hlp.resampling(w)
				
				Xprev[i,:,:] = Xprev[i,ancestors,:]
				Xcur[i,:,:j+1] = Xcur[i,ancestors,:j+1]
				
		# Outer resampling
		maxLogZ = np.max(logZ)
		outerW = np.exp(logZ - maxLogZ)
		outerW /= np.sum(outerW)
		outerAncestors = hlp.resampling(outerW)
		Xprev = Xcur[outerAncestors,:,:]
		
		ESS[t] = 1/np.sum(outerW**2)
		f = open(filename, 'a')
		tmpVec = np.r_[t+1, ESS[t], np.mean(np.mean(Xprev,axis=0),axis=0), np.mean(np.mean(Xprev**2,axis=0),axis=0)]
		np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
		f.close()
