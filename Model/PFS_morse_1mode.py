import numpy as np
from numpy import exp
from numpy import array as A
from numpy import diag_indices as Dii
from scipy.special import assoc_laguerre as Lg
from math import factorial as f
 

def ĉ(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

class parameters():
   dtN = 15.0
   NSteps = int(12000/dtN)
   NTraj = 10000
   
   
   NStates = 10
   M = 1.0
   nskip = 1
   ndof = 1
   
   ωc   = 0.05/27.2114
   
   
   β = 1052.59 #beta for sampling (here 300K)
   χ = 0.5* ωc

   dtE = dtN/12
   initState = 0
   λ = 0.02/27.2114
   #------------------------------------------------------------------ 

def μ(R):
    return (1/1836**0.5) * R
def dμ(R):
    return (1/1836**0.5) 

def Hel(R):
    NStates = parameters.NStates
    ωc      = parameters.ωc
    Vij = np.zeros((NStates,NStates))

    # Diagonal only
    Vij[Dii(NStates)] =  ωc * np.arange(NStates)

    return Vij



def dHel0(R):
    a = 0.0000000027
    b = 0.000006#0.000003
    # a * X^4 - b * X^2 
    # 4 * a * X^3 - 2 * b * X                                                                         
    return 4.0 * a * R**3 - 2 * b * R

def dHel(R):
    NStates = parameters.NStates
    χ       = parameters.χ
    dVij = np.zeros((NStates,NStates,1))
    ωc   = parameters.ωc
    a  = ĉ(NStates) 
    dij =  -(a.T - a) * dµ(R) * χ /ωc

    Hij =  Hel(R)
    dVij[:,:,0] = dij @ Hij - Hij @ dij 

    return dVij




# <n(R2)|m(R1)>
# https://link.springer.com/content/pdf/10.1007%2F978-1-4020-5796-0_4.pdf
def Snm(R1,R2):
    ωc  = parameters.ωc
    NStates = parameters.NStates
    χ   = parameters.χ
    Dµ  = -(µ(R2) - µ(R1))

    dZP = (1/(2*ωc)) ** 0.5
    qc0 = Dµ * χ * (2/ωc**3.0)**0.5 
    ξ   = qc0/(4.0 * dZP) # l/w0
    # no need to recalculate
    G   = np.exp(-2*ξ**2.0)
    X   = (2.0*ξ)**2.0
    # + at 0 PRB.72.195410
    Oij = np.zeros((NStates,NStates))
    for n in range(NStates):
        Oij[n,n] = G * Lg(X,n,0) 
        for m in range(n+1,NStates):
            Oij[m,n] = G * (-ξ * 2)**(m-n) * (f(n)/f(m))**0.5 * Lg(X,n,m-n) 
            Oij[n,m] = Oij[m,n] / (-1) ** (n-m) 

    return Oij



def initR():
    a = 0.0000000027
    b = 0.000006#0.000003
    R0 = - ((b/(2*a))**0.5)
    P0 = 0.0
    M = parameters.M
    ω = 2 * (b)**0.5
    β = parameters.β
    

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω


    R = R0  + np.random.normal()*sigR 
    P = P0  + np.random.normal()*sigP  

    return np.array([R]), np.array([P])

 
