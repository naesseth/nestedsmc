import helpfunctions as hlp
import numpy as np
import nestedFAPF2 as nsmc
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", type=int, help="State dimension")
parser.add_option("--tauPhi", type=float, help="Measurement precision")
parser.add_option("-N", type=int, help="Particles")
parser.add_option("--nrRuns", type=int, help="Number of independent runs")
(args, options) = parser.parse_args()

d=args.d
a = 0.5
tauPsi = 1.
tauRho = 1.
tauPhi = args.tauPhi
filename = 'simulatedData/d'+str(d)+'tauPhi'+str(tauPhi)+'y.txt'
y = np.loadtxt(filename)
filename = 'simulatedData/d'+str(d)+'tauPhi'+str(tauPhi)+'P.txt'
P = np.loadtxt(filename)
P = P[:d,:d]

T = y.shape[1]


N = args.N

for j in range(args.nrRuns):
	filename = './results/paper/d'+str(d)+'_N'+str(N)+'tauPhi'+str(tauPhi)+'_bootstraprun'+str(j+1)+'.csv'
	f = open(filename, 'w')
	f.close()
	
	xCur = np.zeros((N,d))
	xPrev = np.zeros((N,d))
	ancestors = np.zeros(N)
	weights = np.zeros(N)
	logWeights = np.zeros(N)
	ESS = np.zeros(T)

	xCur = np.random.multivariate_normal(np.zeros(d), P, size=N)
	xPrev = xCur

	# t = 1
	for i in range(N):
		logWeights[i] = -0.5*tauPhi*np.sum((xCur[i,:]-y[:,0])**2)
	maxLw = np.max(logWeights)
	weights = np.exp(logWeights-maxLw)
	weights /= np.sum(weights)
	ancestors = hlp.resampling(weights)
	xCur = xCur[ancestors,:]
	ESS[0] = 1/np.sum(weights**2)

	f = open(filename, 'a')
	tmpVec = np.r_[1, ESS[0], np.mean(xCur,axis=0), np.mean(xCur**2,axis=0)]
	np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
	f.close()
	   
	# t > 1 
	for t in np.arange(1,T):
		# Resampling
		ancestors = hlp.resampling(weights)
		
		# Generate samples
		rndSamp = np.random.multivariate_normal(np.zeros(d), P, size=N)
		
		# Propagate
		for i in range(N):
			mu = tauRho*a*np.dot(P,xPrev[ancestors[i],:])
			xCur[i,:] = mu + rndSamp[i,:]
			logWeights[i] = -0.5*tauPhi*np.sum((xCur[i,:]-y[:,t])**2)
		maxLw = np.max(logWeights)
		weights = np.exp(logWeights-maxLw)
		weights /= np.sum(weights)
		ESS[t] = 1/np.sum(weights**2)
		xPrev = xCur
		
		f = open(filename, 'a')
		tmpVec = np.r_[t+1, ESS[t], np.mean(xCur,axis=0), np.mean(xCur**2,axis=0)]
		np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
		f.close()

