import numpy as np
import sys, os
from numpy.random import normal as gran
from numpy import diag_indices as ii

import random

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "focused"):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == "focused"):
        qF[initState] = 1.0
        qB[initState] = 1.0
        pF[initState] = 1.0
        pB[initState] = -1.0 # This minus sign allows for backward motion of fictitious oscillator
    elif (stype == "sampled"):
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 


def Umapii(qF, qB, pF, pB, dtN, ωc):
    # see https://physics.info/sho/
    ω = ωc * np.arange(len(qF))
    # vB, vF = pB/m, pF/m
    
    sinωt = np.sin(ω*dtN)
    cosωt = np.cos(ω*dtN)
    qF0, qB0, pF0, pB0 = qF * 1.0, qB * 1.0, pF * 1.0, pB * 1.0 

    qF = qF0 * cosωt + pF0 * sinωt
    qB = qB0 * cosωt + pB0 * sinωt 

    pF = -qF0 * sinωt + pF0 * cosωt
    pB = -qB0 * sinωt + pB0 * cosωt

    return qF, qB, pF, pB



def Force(dat):

    dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat.dH0
    qF, pF, qB, pB =  dat.qF, dat.pF, dat.qB, dat.pB
    # F = np.zeros((len(dat.R)))
    F = -dH0
    for i in range(len(qF)):
        F -= 0.25 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 + qB[i] ** 2 + pB[i] ** 2)
        for j in range(i+1, len(qF)):
            F -= 0.5 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def ForceF(dat):

    dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat.dH0
    qF, pF, qB, pB =  dat.qF, dat.pF, dat.qB, dat.pB
    # F = np.zeros((len(dat.R)))
    F = -dH0
    for i in range(0,len(qF)-1):
        #F -= 0.25 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 + qB[i] ** 2 + pB[i] ** 2)
        #for j in range(i+1, len(qF)):
        j = i+1
        F -= 0.5 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def VelVer(dat) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
    R1   = dat.R * 1.0
    # data 
    qF, qB, pF, pB = dat.qF * 1.0, dat.qB *  1.0, dat.pF * 1.0, dat.pB * 1.0
    par =  dat.param
    v = dat.P/par.M
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep
    
    # half-step mapping
    qF, qB, pF, pB = Umapii(qF, qB, pF, pB, par.dtN/2.0, dat.Hij[ii(len(qF))])
    dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    """
    for t in range(int(np.floor(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, dat.Hij)
    dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    """
    # ======= Nuclear Block ==================================
    F1    =  Force(dat) # force with {qF(t+dt/2)} * dH(R(t))
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Compute Overlaps -----
    R2   = dat.R * 1.0 
    Sij  = par.Snm(R1,R2)

    #------ Transform Basis  -----
    qF, qB, pF, pB = Sij @ qF, Sij @ qB, Sij @ pF, Sij @ pB
    dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat.P = v * par.M
    # =======================================================
    
    # half-step mapping
    dat.Hij = par.Hel(dat.R) # do QM
        
    qF, qB, pF, pB = Umapii(qF, qB, pF, pB, par.dtN/2.0, dat.Hij[ii(len(qF))])
    dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    """
    for t in range(int(np.ceil(EStep/2))):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, dat.Hij)
    dat.qF, dat.qB, dat.pF, dat.pB = qF, qB, pF, pB 
    """
    return dat

def VelVerL(dat) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
    par =  dat.param
    ωc  =  par.ωc
    R1   = dat.R * 1.0
    ndof = len(R1)
    # Langevin Stuff 
    λ    = par.λ
    β  = par.β
    σ = (2.0 * λ/(β * par.M )) ** 0.5
    ξ = gran(0, 1, ndof)   
    θ = gran(0, 1, ndof)  
    C = 0.28867513459


    #-------------------------
    # data 
    #qF, qB, pF, pB = dat.qF * 1.0, dat.qB *  1.0, dat.pF * 1.0, dat.pB * 1.0
    
    v = dat.P/par.M


    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep
    
    # half-step mapping
    dat.qF, dat.qB, dat.pF, dat.pB = Umapii(dat.qF, dat.qB, dat.pF, dat.pB, par.dtN/2.0, ωc)
    #dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    # ======= Nuclear Block ==================================
    F1    =  dat.F1 #Force(dat) # force with {qF(t+dt/2)} * dH(R(t))

    # 
    Af = (0.5 * par.dtN**2) * (F1/par.M - λ * v) + (σ * par.dtN**(3.0/2.0)) * (0.5 * ξ + C * θ) 

    # Propagate R
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M +  Af 
    
    #------ Compute Overlaps -----
    R2   = dat.R * 1.0 
    Sij  = par.Snm(R1,R2)

    #------ Transform Basis  -----
    #qF, qB, pF, pB = Sij @ qF, Sij @ qB, Sij @ pF, Sij @ pB
    #dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    dat.qF, dat.qB, dat.pF, dat.pB = Sij @ dat.qF, Sij @ dat.qB, Sij @ dat.pF, Sij @ dat.pB
    #------ Do QM ----------------
    #dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = ForceF(dat) # force with {qF(t+dt/2)} * dH(R(t+ dt))

    v += 0.5 * (F1 + F2) * par.dtN / par.M - par.dtN * λ * v +  σ * (par.dtN**0.5) * ξ - Af * λ

    dat.P = v * par.M
    # =======================================================
    
    # half-step mapping
    dat.qF, dat.qB, dat.pF, dat.pB = Umapii(dat.qF, dat.qB, dat.pF, dat.pB, par.dtN/2.0, ωc)
    #dat.qF, dat.qB, dat.pF, dat.pB = qF * 1, qB * 1, pF * 1, pB * 1 
    #dat.qF, dat.qB, dat.pF, dat.pB = Sij @ dat.qF, Sij @ dat.qB, Sij @ dat.pF, Sij @ dat.pB

    # save force for next step
    dat.F1 = F2
    return dat

def pop(dat):
    return np.outer(dat.qF + 1j * dat.pF, dat.qB-1j*dat.pB) * dat.rho0

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    # Thermalize Init State
    β     = parameters.β
    ωc    = parameters.ωc
    ρE    = np.exp(-β * np.arange(NStates) * ωc)
    ρE    = ρE/np.sum(ρE)
    print (f"ρE : {ρE}")
    ρES   = np.array([np.sum(ρE[:i+1]) for i in range(NStates)])
    Ψ     = np.zeros((NStates))
    #--------------------------
    stype = parameters.stype
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    fs = np.zeros((len(parameters.initR()[0]),NSteps//nskip + pl), dtype=complex)

    reaction = np.zeros((2,NSteps//nskip + pl))

    # Ensemble
    for itraj in range(NTraj): 
        s1    = ρES- np.random.random()
        for i in range(NStates):
            if (s1[i]>0):
                initState = i
                break
        Ψ[initState] += 1
        print (f"|Ψ> = |{initState}> | Traj: {itraj}")

        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()

        # set propagator
        vv  = VelVerL

        # Call function to initialize mapping variables
        dat.qF, dat.qB, dat.pF, dat.pB = initMapping(NStates, initState, stype) 

        # Set initial values of fictitious oscillator variables for future use
        qF0, qB0, pF0, pB0 = dat.qF[initState], dat.qB[initState], dat.pF[initState], dat.pB[initState] 
        dat.rho0 = 0.25 * (qF0 - 1j*pF0) * (qB0 + 1j*pB0)

        #----- Initial QM --------
        # dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        dat.F1    =  Force(dat)
        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                fs[:,iskip] += dat.R[:]
                reaction[:, iskip]  += 1*(dat.R[0]>0.0),  1*(dat.R[0]<0.0)
                iskip += 1
            #-------------------------------------------------------
            
            dat =  vv(dat)
    
    print (f"ρE (theory)     = {ρE}")
    print (f"ρE (simulation) = {Ψ/np.sum(Ψ)}")   

    return rho_ensemble, R_ensemble, reaction
 

