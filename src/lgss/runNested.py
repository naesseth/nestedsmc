import helpfunctions as hlp
import numpy as np
import nestedFSMC as nsmc
from optparse import OptionParser



def runNested(d, tauPhi, N, M, nrRuns):
    # Model init
    a = 0.5
    tauPsi = 1.
    tauRho = 1.
    params = hlp.params(a = a,tauPsi = tauPsi,tauRho = tauRho,tauPhi = tauPhi)

    filename = 'simulatedData/d'+str(d)+'tauPhi'+str(tauPhi)+'y.txt'
    y = np.loadtxt(filename)
    T = y.shape[1]

    for j in range(nrRuns):
        # Algorithm init
        logZ = np.zeros(N)
        ancestors = np.zeros(N)
        X = np.zeros((N,d))
        ESS = np.zeros(T)

        # Setup new file
        filename = './results/paper/d'+str(d)+'_N'+str(N)+'_M'+str(M)+'tauPhi'+str(tauPhi)+'_nestedSMCrun'+str(j+1)+'.csv'
        f = open(filename, 'w')
        #f.write('Iter ESS E[x] E[x**2]\n')
        f.close()
        
        for t in range(T):
            q = [ nsmc.nestedFAPF(params, y[:,t], M, X[i,:]) for i in range(N) ]

            for i in range(N):
                logZ[i] = q[i].logZ
                
            maxLz = np.max(logZ)
            w = np.exp(logZ-maxLz)
            w /= np.sum(w)
            ESS[t] = 1/np.sum(w**2)
            ancestors = hlp.resampling(w)
            
            for i in range(N):
                X[i,:] = q[ancestors[i]].simulate(1,BS=True)    
            
            f = open(filename, 'a')
            tmpVec = np.r_[t+1, ESS[t], np.mean(X,axis=0), np.mean(X**2,axis=0)]
            np.savetxt(f, tmpVec.reshape((1,len(tmpVec))),delimiter=',')
		f.close()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", type=int, help="State dimension")
    parser.add_option("--tauPhi", type=float, help="Measurement precision")
    parser.add_option("-N", type=int, help="Particles")
    parser.add_option("-M", type=int, help="Nested particles")
    parser.add_option("--nrRuns", type=int, help="Number of independent runs")
    (args, options) = parser.parse_args()
    runNested(args.d, args.tauPhi, args.N, args.M, args.nrRuns)
