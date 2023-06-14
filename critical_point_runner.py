import sys
import critical_point

lambda1 = float(sys.argv[1])
mu0 = float(sys.argv[2])
mu1 = float(sys.argv[3])
mu2 = float(sys.argv[4])
ml = float(sys.argv[5])
tmin = float(sys.argv[6])
tmax = float(sys.argv[7])
numtemp = int(sys.argv[8])
minsigma = float(sys.argv[9])
maxsigma = float(sys.argv[10])
mu_initial = float(sys.argv[11])
delta_mu = float(sys.argv[12])
mu_precision = int(sys.argv[13])

critical_point.critical_point_refined(lambda1,mu0,mu1,mu2,ml,tmin,tmax,numtemp,minsigma,maxsigma,mu_initial,delta_mu,mu_precision)
