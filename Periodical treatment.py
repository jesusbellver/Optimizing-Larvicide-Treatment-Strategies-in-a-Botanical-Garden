if __name__ == '__main__':

    #Import
    import numpy as np
    import math
    import random
    import matplotlib.pyplot as plt
    import scipy.integrate as integrate
    from scipy.integrate import odeint
    from scipy.integrate import solve_ivp
    from scipy.optimize import root_scalar
    from scipy.optimize import minimize
    from copy import deepcopy as dc
    import time
    import seaborn
    import pickle
    from scipy.interpolate import PchipInterpolator
    #import multiprocessing as mp  #to parallelize
    import os
    import datetime
    from joblib import Parallel, delayed

    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Get the directory path of the script
    directory_path = os.path.dirname(script_path)


    #LOAD DATA FROM FILES
    Year=2022  #year for which you load data

    #Load biological parameters
    # Birth rate
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/Alpha.pkl'),  'rb') as f:
        Alpha = pickle.load(f)
    # Development rate
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/Mu.pkl'),  'rb') as f:
        Mu = pickle.load(f)
    # Load Dev rate + Larval death rate
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/DeltaL.pkl'),  'rb') as f:
        DeltaL = pickle.load(f)
    # Adult death rate
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/DeltaA.pkl'),  'rb') as f:
        DeltaA = pickle.load(f)



    # Load M for Embornals
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/M.pkl'),  'rb') as f:
        M = pickle.load(f)

    # Load M for traps
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/M_Trap.pkl'), 'rb') as f:
        M_Trap = pickle.load(f)

    M_Trap=M_Trap.T


    # Load Vols
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/Vols.pkl'), 'rb') as f:
        Vols_temp = pickle.load(f)

    # Load K
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/K.pkl'), 'rb') as f:
        K_temp = pickle.load(f)


    S=len(Vols_temp)
    Range_S = range(S)

    def Vols(t):
        return np.array([Vols_temp[i](t) for i in Range_S])

    def K(t):
        return np.array([K_temp[i](t) for i in Range_S])


    def G(t, y):
        L=y[0:S]
        A = y[S:S2]
        c = y[S2:S3]

        dLdt = Alpha(t) * A  - (Mu(t)+DeltaL(t))*L - tau * np.divide(c*L,Vols(t)) - np.divide(L*L,K(t))
        dAdt = 0.5 * Mu(t) * L - DeltaA(t) * A + M.dot(A)  #1/2 por que sólo nos interesan female adults
        dcdt = - kc * c - kL*np.divide(c*L,Vols(t))

        return np.concatenate((dLdt,dAdt,dcdt))


    def system_solve(y0, Gr_dose, tk, ck, t0, T):

        sol=solve_ivp(G, [t0,tk[0]], y0, method=Method, rtol=r_tol)
        t=sol.t
        Y=sol.y

        for k in Range_NJ_1:
            if T>tk[k]:
                Y[S2:S3,-1] = Y[S2:S3,-1] + ck[k]*Gr_dose #We add treatment in the breeding sites
                #Y[0:S,-1] = np.maximum(Y[0:S,-1], 0.25)

                sol=solve_ivp(G, [tk[k],tk[k+1]], Y[:,-1], method=Method, rtol=r_tol)
                t = np.concatenate((t,sol.t))
                Y = np.concatenate((Y, sol.y),axis=1)

        if T>tk[NJ-1]:
            Y[S2:S3,-1] = Y[S2:S3,-1] + ck[-1]*Gr_dose
            #Y[0:S,-1]=np.maximum(Y[0:S,-1],0.25)
            sol=solve_ivp(G, [tk[-1],T], Y[:,-1], method=Method, rtol=r_tol)
            t=np.concatenate((t,sol.t))
            Y = np.concatenate((Y, sol.y),axis=1)

        return t,Y

    #Objective function
    def crit(Y,t):
        return integrate.trapz(sum(Y[S:S2]), t) #INT A(t) dt

    #Treatment at each breeding site. From Embornal_Clustering.py
    Gr_dose = np.array([10,10,10,10,10,30,30,10,10,10,10,10,10,10,10,10,10,10,10,20,20,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,20,10,10,20,10,10,10,20,10,10,10,10,10,10,10,10,10,20,10,20,10,10,10,10,10,10])  # Por promedio tratamiento reales Santi, 20,30,45. Real es 18.7, no 20

    #Constraint
    def ck_Cost(ck,Gr_dose):
        return sum(sum(Gr_dose*ck))

    #RANDOMIZED STARTING POINTS
    def generate_tk(t_ini, t_fin, min_d,NJ):
        # Initialize the first number in the list
        nums = [t_ini+ (t_fin-t_ini-(NJ-1)*min_d)*random.random()] #generate NJ treatments equally spaced that fit in [t_ini,t_fin]
        nums.extend([nums[0]+j*min_d for j in range(1,NJ)])
        return np.array(nums)


    def generate_ck(C, Gr_dose, S, NJ):  #at random
        ck = np.array([1 for n in range(NJ*S)]) #Generate all ones

        while ck_Cost(ck.reshape(NJ,S), Gr_dose) > C:
            ones = [i for i, x in enumerate(ck) if x == 1]  #turn the 1 to 0 to satisfy the constraint
            i = random.choice(ones)
            ck[i] = 0

        ck = ck.reshape(NJ, S) #Estructura ck=[[ S] NJ]

        return ck

    def greedy_update(arr1, arr2):
        # Combine the two arrays into a list of tuples (value, index)
        combined = list(enumerate(zip(arr1, arr2)))

        # Sort the list based on the values in the first array in reverse order
        combined.sort(key=lambda x: x[1][0], reverse=True)

        # Iterate through the sorted list
        for original_index, val in combined:
            if arr2[original_index] == 0:
                arr2[original_index] = 1
                break

        return arr2
    def generate_ck_greedy(C, Gr_dose, S, NJ, y0, tk, t0, T):  #Select mos populated breeding sites
        ck0 = np.array([0 for n in range(NJ*S)]) #Generate all ones
        ck0 = ck0.reshape(NJ, S)

        for j in range(NJ):
            t, Y = system_solve(y0, Gr_dose, tk, ck0, t0, tk[j])
            ck = dc(ck0)
            print(Y[0:S,-1])
            while True:
                ck[j] = greedy_update(Y[0:S,-1], ck[j])
                if ck_Cost(ck, Gr_dose) <= (j+1) * C / 4:
                    ck0 = dc(ck)
                else:
                    break
            print(ck0,ck_Cost(ck0,Gr_dose))

        return ck0




    #SIMULATION
    def simulation_tkck(C, t0, T, t_ini, t_fin, min_d, S, NJ, N_sims, filename_tkck):

        All_tkckJtY = []

        for n in range(N_sims):
            t01=time.time()
            print('\n Simulation:', n + 1, flush=True)


            L0 = np.array([cst_L * K(t0)[i] * (0.5 + 0.5 * random.random()) for i in Range_S])
            A0 = np.array([cst_A * K(t0)[i] * (0.5 + 0.5 * random.random()) for i in Range_S])
            c0 = np.array([0 for i in Range_S])
            y0=np.concatenate((L0,A0,c0))

            #Para la primera ejecución
            tk = generate_tk(t_ini, t_fin, min_d, NJ)
            #ck = generate_ck(C, Gr_dose, S, NJ)
            ck =generate_ck_greedy(C, Gr_dose, S, NJ, y0, tk, t0, T)
            print(ck_Cost(ck, Gr_dose), flush=True)

            t, Y = system_solve(y0, Gr_dose, tk, ck, t0, T)

            All_tkckJtY.append([dc(tk),dc(ck),dc(crit(Y, t)),dc(t),dc(Y),dc(ck_Cost(ck,Gr_dose))])

            t02=time.time()
            print(tk,ck,crit(Y,t))
            print('\n t of Simulation:', t02 - t01, flush=True)

            with open(os.path.join(directory_path, 'Results', f'{Year}/Periodic', filename_tkck), 'wb') as f:
                pickle.dump(All_tkckJtY, f)

        return All_tkckJtY





    #BIOLOGICAL PARAMETERS
    #Computed from Parameters_MRR_new_weibull
    sigma = 12.59  # 12.5875   de weighted average de los dos MRR tiene 315 mosq. y Lugano 33
    alpha_w = 1.323  # shape
    beta_w = 1 / 110.3  # scale


    # fit n=100000. Del .py "Parameter barreños Datos Separados Cantidad".
    #tau= 407.6573600454137
    tau= 407.66
    #kc= 0.08904292214328913
    kc=0.089
    #kL= 0.021593981164619105
    kL= 0.021

    #gamma_dose = Fabricante: 10g/50l Santi: 39.75g/50l~40g/50l

    # chi= 0.01  # Trap efficacy
    # #s_r= (39+116+281+817+770+494+34)/(39+116+281+817+770+494+34+5+19+41+177+176+55+1)  #capture sex ratio. F/(F+M)
    # s_r=0.843

    #Time PARAMETERS
    t0 = 0
    if Year==2020:
        t_ini = 92  # 1 de Abril
        t_fin = 305 #31 de Octubre
        T = 366
    else:
        t_ini = 91  # 1 de Abril
        t_fin = 304 #31 de Octubre
        T = 365

    S2 = int(2 * S)
    S3 = int(3 * S)
    ST = len(M_Trap)
    Range_ST=range(ST)

    NJ = 4 #Number of treatments
    Range_NJ=range(NJ)
    Range_NJ_1=range(NJ-1)
    Range_1_NJ=range(1,NJ)

    C = 810  #Amount of treatment available. Max= 3240, C75=2430, C50=1720, C25=810

    #Optimization parameters
    N_sims = 25  # número de simulaciones

    min_d = 7*7  #treatments spaced 7 weeks (7*7 days). Because of Abundances Laura

    #Method parameters
    Method = 'Radau'
    r_tol = 1e-4   #for C=3240, 1e-9, C=2430, 1e-6, C=2620, e-5, C=810, e-4
    cst_L = 0.25
    cst_A = cst_L * Mu(t0) / (2 * DeltaA(t0))


    # # # # # #RESULTS
    filename_tkck = f'All_tkckJtY_Periodic_Greedy_{Year}_{C}.pkl'

    t1 = time.time()

    All_tkckJtY = simulation_tkck(C, t0, T, t_ini, t_fin, min_d, S, NJ, N_sims, filename_tkck)

    with open(os.path.join(directory_path, 'Results', f'{Year}/Periodic', filename_tkck), 'wb') as f:
        pickle.dump(All_tkckJtY, f)


    t2=time.time()

    print(t2-t1)


# #
# # #Pruebas
# #tk Santi
# tk= [102, 168, 218, 270]
#
# chi = 0.01
# tau=407.66
#
#
#
# #Santi Groups dose.
# #ck=np.array([[0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1],[0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1],[0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1]])
# ##Santi real
# ## ck=np.array([[0,10,20,20,10,20,10,20,0,20,10,20,10,0,20,20,20,10,10,10,10,10,10,20,20,20,20,0,5,10,0,0,0,0,10],[0,10,20,20,10,20,20,40,0,30,10,40,10,10,30,20,40,10,10,10,10,20,20,0,20,20,40,40,0,10,20,15,20,10,10],[0,10,40,20,10,20,20,40,0,30,10,0,10,10,30,20,40,10,10,10,10,20,40,20,20,20,40,40,0,0,20,15,20,0,10],[0,10,100,20,20,30,10,60,0,40,10,40,10,10,30,20,40,20,20,10,10,20,10,0,0,20,60,40,10,10,30,20,0,10,20]])
#
#
# #
# r_tol=5e-8
# t1 = time.time()
# L0 = np.array([cst_L*K(t0)[i]*0.75 for i in Range_S])
# A0 = np.array([cst_A*K(t0)[i]*0.75 for i in Range_S])
# c0 = np.array([0 for i in Range_S])
# y0=np.concatenate((L0,A0,c0))
#
#
# # # tk=[ 94.14219363 ,133.51988809 ,163.61256478 ,256.49623993]
# ck = generate_ck(2340, Gr_dose, S, NJ)
# #tk= [102, 168, 218, 270]
# tk=[t_ini+j*(t_fin-t_ini)/5 for j in range(1,5)]
# NJ=len(tk)
# Range_NJ = range(NJ)
# Range_NJ_1 = range(NJ - 1)
# Range_1_NJ = range(1, NJ)
#
#
# #
# t, Y = system_solve(y0, Gr_dose, tk, ck, t0, T)
#
# J = crit(Y, t)
#
# print(J)
#
# with open(f'Results/{Year}/Controlled.pkl', 'wb') as f:
#     pickle.dump((J,t,Y), f)
#
# t2=time.time()
# print(t2-t1)
#
#
# print(ck_Cost(Gr_dose,ck))
# # print(t2-t1)
# #
#
# for i in range(20):
#     r_tol=1e-8
#     t1 = time.time()
#     L0 = np.array([cst_L*K(t0)[i]*(0.5+0.5*random.random()) for i in Range_S])
#     A0 = np.array([cst_A*K(t0)[i]*(0.5+0.5*random.random()) for i in Range_S])
#     c0 = np.array([0 for i in Range_S])
#     y0=np.concatenate((L0,A0,c0))
#
#
#     tk = generate_tk(t_ini, t_fin, min_d, NJ)
#     ck = generate_ck(2430, Gr_dose, S, NJ)
#     ck = np.array([[1 for i in Range_S] for j in Range_NJ])
#     print(ck_Cost(ck,Gr_dose))
#
#     t,Y = system_solve(y0, Gr_dose, tk, ck, t0, T)
#
#     J=crit(Y,t)
#
#     t2 = time.time()
#
#     print('tk',tk,flush=True)
#     print(J,flush=True)
#     print('time',t2-t1,flush=True)
#
