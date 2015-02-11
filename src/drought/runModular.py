import nestedSMCmodular as nsmc
import helpfunctions as hlp
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--tStart", type=int, help="Year to start")
parser.add_option("--tEnd", type=int, help="Year to end")
parser.add_option("--Np", type=int, help="1st level particles")
parser.add_option("-N", type=int, help="2nd level particles")
parser.add_option("-M", type=int, help="3rd level particles")
(args, options) = parser.parse_args()

# Model init
tStart = args.tStart
tEnd = args.tEnd
Np = args.Np
N = args.N
M = args.M
region = 'us'
#region = 'sahel'
if region == 'us':
	X = np.zeros((Np,20,30))
	xC = np.zeros((20,30),dtype=bool)
else:
	X = np.zeros((Np,24,44))
	xC = np.zeros((24,44),dtype=bool)
logZ = np.zeros(Np)


q = []

for i in range(Np):
    q.append(nsmc.nestedSMC(t=tStart,N=N,M=M,xCond=xC))
    logZ[i] = q[i].logZ
maxLZ = np.max(logZ)
w = np.exp(logZ-maxLZ)
w /= np.sum(w)
ESS = 1/np.sum(w**2)
ancestors = hlp.resampling(w)
for i in range(Np):
    X[i,:,:] = q[ancestors[i]].simulate()

folder = '/data/chran60/nestedsmc/drought'
np.savetxt(folder+'/Np'+str(Np)+'_M'+str(N)+'_M'+str(M)+'_t'+str(tStart)+region+'.csv',np.mean(X,axis=0),delimiter=',')

for t in np.arange(1,tEnd-tStart+1):
    print 't: ',tStart+t, ' ESS: ', ESS
    q = []
    for i in range(Np):
        q.append(nsmc.nestedSMC(t=t+tStart,N=N,M=M,xCond=X[i,:,:]))
        logZ[i] = q[i].logZ
    maxLZ = np.max(logZ)
    w = np.exp(logZ-maxLZ)
    w /= np.sum(w)
    ESS = 1/np.sum(w**2)
    ancestors = hlp.resampling(w)
    for i in range(Np):
        X[i,:,:] = q[ancestors[i]].simulate()
    np.savetxt(folder+'/Np'+str(Np)+'_M'+str(N)+'_M'+str(M)+'_t'+str(tStart+t)+region+'.csv',np.mean(X,axis=0),delimiter=',')

