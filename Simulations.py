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
    Year=2021  #year for which you load data

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

    #Load Group dose.  #Dose of treatment at each breeding site. From Embornal_Clustering.py
    with open(os.path.join(directory_path, f'Parameters_K_A_V/{Year}/GroupsDose.pkl'), 'rb') as f:
        Gr_dose = pickle.load(f)

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
            Y[S2:S3,-1] = Y[S2:S3,-1] + ck[k]*Gr_dose #We add treatment in the breeding sites
            #Y[0:S,-1] = np.maximum(Y[0:S,-1], 0.25)

            sol=solve_ivp(G, [tk[k],tk[k+1]], Y[:,-1], method=Method, rtol=r_tol)
            t = np.concatenate((t,sol.t))
            Y = np.concatenate((Y, sol.y),axis=1)

        Y[S2:S3,-1] = Y[S2:S3,-1] + ck[-1]*Gr_dose
        #Y[0:S,-1]=np.maximum(Y[0:S,-1],0.25)
        sol=solve_ivp(G, [tk[-1],T], Y[:,-1], method=Method, rtol=r_tol)
        t=np.concatenate((t,sol.t))
        Y = np.concatenate((Y, sol.y),axis=1)

        return t,Y

    #Objective function
    def crit(Y,t):
        return integrate.trapz(sum(Y[S:S2]), t) #INT A(t) dt


    #Constraint
    def ck_Cost(ck,Gr_dose):
        return sum(sum(Gr_dose*ck))

    #RANDOMIZED STARTING POINTS
    def generate_tk(t_ini, t_fin, min_d,NJ):
        # Initialize the first number in the list
        nums = [t_ini+ (t_fin-t_ini) * random.random()]
        for i in range(1, NJ):
            # Generate a new random number within the specified range
            new_num = t_ini+ (t_fin-t_ini)*random.random()
            while any([abs(new_num - n) <= min_d for n in nums]):
                # Keep generating new numbers until the minimum spacing is satisfied
                new_num = t_ini+ (t_fin-t_ini) * random.random()
            nums.append(new_num)

        return np.array(sorted(nums))

    def generate_ck(C, Gr_dose, S, NJ):
        ck = np.array([1 for n in range(NJ*S)]) #Generate all ones

        while ck_Cost(ck.reshape(NJ,S), Gr_dose) > C:
            ones = [i for i, x in enumerate(ck) if x == 1]  #turn the 1 to 0 to satisfy the constraint
            i = random.choice(ones)
            ck[i] = 0

        ck = ck.reshape(NJ, S) #Estructura ck=[[ S] NJ]

        return ck


    def explore_tk_ck(args):

        C,y0, tk, ck, t0, T, t_ini, t_fin, min_d, S, NJ, width_tk_search ,depth_ck_search = args

        tkmins= np.array([max(tk[0]-width_tk_search,t_ini)]+[max(tk[k]-width_tk_search,tk[k-1]+min_d) for k in Range_1_NJ]) #mínimos valores que puede tomar tk. Componente a componente
        tkmaxs= np.array([min(tk[k]+width_tk_search,tk[k+1]-min_d) for k in Range_NJ_1]+[min(tk[-1]+width_tk_search,t_fin)]) #máximos valores que puede tomar tk. Componente a componente
        tk0=dc(tkmins+np.array([random.random() for k in Range_NJ])*(tkmaxs-tkmins))

        #Para optimizar sólo tk
        #ck0=dc(ck)

        # Para optimizar tk y ck
        ck0 = dc(ck.reshape(1, S * NJ)[0])
        zeros = [i for i, x in enumerate(ck0) if x == 0]
        for j in range(int(np.round(depth_ck_search*(1+random.random())/2))):  #eliminar aleatoriamente entre depth_ck_search/2 y depth_ck_search tratamientos
            if len(zeros)>0: #in case all are ones already
                i = random.choice(zeros)
                ck0[i] = 1

        while ck_Cost(ck0.reshape(NJ, S), Gr_dose) > C:
            ones = [i for i, x in enumerate(ck0) if x == 1]
            i = random.choice(ones)
            ck0[i] = 0

        ck0 = ck0.reshape(NJ, S)


        #Esto siempre
        t,Y0=system_solve(y0, Gr_dose, tk0, ck0, t0, T)
        J0 = crit(Y0,t)
        cost= ck_Cost(ck0,Gr_dose)

        print('\n', J0, tk0, ck0, cost, flush=True)

        return tk0, ck0, J0, t, Y0, cost



    #SIMULATION
    def simulation_tkck(C, t0, T, t_ini, t_fin, min_d, S, NJ, N_sims, N_iter, width_tk_search, depth_ck_search,filename_tkck,filename_J_tracker):

        All_tkckJtY = []
        All_J_tracker = []

        for n in range(N_sims):
            print('\n Simulation:', n + 1, flush=True)


            L0 = np.array([cst_L * K(t0)[i] * (0.5 + 0.5 * random.random()) for i in Range_S])
            A0 = np.array([cst_A * K(t0)[i] * (0.5 + 0.5 * random.random()) for i in Range_S])
            c0 = np.array([0 for i in Range_S])
            y0=np.concatenate((L0,A0,c0))

            #Para la primera ejecución
            # tk = generate_tk(t_ini, t_fin, min_d, NJ)
            # #ck = generate_ck(C, Gr_dose, S, NJ)
            # ck = np.array([[1 for i in Range_S] for j in Range_NJ])   #para opt. solo tk
            # print(ck_Cost(ck, Gr_dose), flush=True)
            #
            # t, Y = system_solve(y0, Gr_dose, tk, ck, t0, T)


            #Para descartar las peores tk_ck iniciales
            while True:

                tk = generate_tk(t_ini, t_fin, min_d, NJ)
                #ck = np.array([[1 for i in Range_S] for j in Range_NJ])  # para opt. solo tk
                ck = generate_ck(C, Gr_dose, S, NJ)

                t, Y = system_solve(y0, Gr_dose, tk, ck, t0, T)
                print(crit(Y, t), flush=True)

                # for Year=2020, C=3240, crit(Y,t) < 5e4
                # for Year=2021, C=3240, crit(Y,t) < 250
                # for Year=2022, C=3240, crit(Y,t) < 5e3

                # for Year=2020, C=2430, crit(Y,t) < 1.4e6
                # for Year=2021, C=2430, crit(Y,t) < 6.5e5
                # for Year=2022, C=2430, crit(Y,t) < 1.75e6

                # for Year=2020, C=1620, crit(Y,t) < 3e6
                # for Year=2021, C=1620, crit(Y,t) < 1.5e6
                # for Year=2022, C=1620, crit(Y,t) < 2.35e6

                # for Year=2020, C=810, crit(Y,t) < 4.5e6
                # for Year=2021, C=810, crit(Y,t) < 2.2e6
                # for Year=2022, C=810, crit(Y,t) < 3.75e6

                if crit(Y, t) > 0 and crit(Y,t) < 2.2e6: #Reinicializamos si las condiciones iniciales son muy malas .Además, a veces hay overflow, esto es para evitar generar tk,ck que dan overflow
                   break


            best_tkckJtY = [dc(tk),dc(ck),dc(crit(Y, t)),dc(t),dc(Y),dc(ck_Cost(ck,Gr_dose))]
            J_tracker = [dc(best_tkckJtY[2])]


            print('Initialization', flush=True)
            print('tkck=', best_tkckJtY[0], best_tkckJtY[1], flush=True)
            print('J=', best_tkckJtY[2], flush=True)

            #tk ck optimization
            W=width_tk_search
            D = depth_ck_search
            count = 1
            count_glob = 1
            while count<=Max_iter and count_glob<=N_iter: #reducir el número de iteraciones máximas a medida que avanza la simulación, ya que ya hemos disminuido W

                if count == 1:
                    print('\n Iteration', count_glob, ' (Sim=', n+1, ')', flush=True)
                t01 = time.time()

                # Shared memory data
                shared_data = [C, y0, best_tkckJtY[0], best_tkckJtY[1], t0, T, t_ini, t_fin, min_d, S, NJ, W, D]

                # Parallelize code
                tkckJtY = Parallel(n_jobs=-1)(delayed(explore_tk_ck)(shared_data) for d in Range_Ntkck)  #n_jobs=-1 para usar todos los cores

                tkckJtY.sort(key=lambda x: x[2])
                tkckJtY=[tkckJtY[r] for r in range(len(tkckJtY)) if tkckJtY[r][2]>0] #there are some numerical problems we are not getting into

                if tkckJtY[0][2] < best_tkckJtY[2] and tkckJtY[0][2] > 0:
                    best_tkckJtY = [dc(tkckJtY[0][0]), dc(tkckJtY[0][1]), dc(tkckJtY[0][2]), dc(tkckJtY[0][3]),dc(tkckJtY[0][4]), dc(tkckJtY[0][5])]
                    J_tracker.append(dc(best_tkckJtY[2]))
                    count = 1
                    count_glob = count_glob + 1

                    W = max(1,width_tk_search/ (1.25 ** (count_glob // 4)))  # parte entera de dividir entre 3, cada 3 iteraciones reduce el espacio de búsqueda
                    D = max(1, depth_ck_search - (count_glob // 4))  # parte entera de dividir entre 3, cada 3 iteraciones reduce el espacio de búsqueda to improve chances of getting better ck

                    t02 = time.time()
                    print('\n t of Iteration:', t02 - t01, flush=True)
                    print('\n tk=', best_tkckJtY[0], flush=True)
                    print('ck=', best_tkckJtY[1], flush=True)
                    print('J=', best_tkckJtY[2], flush=True)
                    print('Cost=', best_tkckJtY[5], flush=True)

                    #Para cortar antes de tiempo las simus si no van bien

                    # if count_glob == 10 and best_tkckJtY[2] > 2.25e5:  # 3.5e5 for C=2430-2020, 2.25e5 for C=2430-2022
                    #     break

                    # if count_glob == 10 and best_tkckJtY[2] > 1.05e6:  # 1.5e6  for C=1620-2020, 6.5e5  for C=1620-2021, 1.05e6  for C=1620-2022
                    #     break
                    #
                    # if count_glob == 20 and best_tkckJtY[2]> 7.5e5:   #8.5e5 for C=1620-2020, 4e5 for C=1620-2021, 7.5e5  for C=1620-2022
                    #     break

                    # if count_glob == 35 and best_tkckJtY[2] > 2.3e6:  # 2.3e6 for C=810-2020
                    #     break

                    if count_glob == 40 and best_tkckJtY[2] > 1.29e6:  # 1.29e6 for C=810-2021
                        break

                    # if count_glob == 50 and best_tkckJtY[2] > 1.75e6:  # 1.75e6 for C=810-2022
                    #    break


                else:
                    if count % 2== 1:  #to reduce the search space if no better solution is found
                        W = max(0.1, W / 1.25)
                        D = max(1, D - 1)
                    count = count + 1
                    print('\n', count_glob, count, 'D=', D, 'W=', W, ' (Sim=', n+1, ')', flush=True)


            t03=time.time()
            print('\n t of Simulation:', t03 - t01, flush=True)


            All_tkckJtY.append(best_tkckJtY)
            All_J_tracker.append(J_tracker)

            with open(os.path.join(directory_path, 'Results', filename_tkck), 'wb') as f:
                pickle.dump(All_tkckJtY, f)

            with open(os.path.join(directory_path, 'Results', filename_J_tracker), 'wb') as f:
                pickle.dump(All_J_tracker, f)

        return All_tkckJtY, All_J_tracker


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
    N_sims = 10  # número de simulaciones
    N_iter = 70  # número de máximo de iteraciones tk, ck en cada simulación (GLOBAL)
    Max_iter= 15  #Número de veces que reduce el tamaño de la búsqueda antes de parar (LOCAL)

    N_tk_ck_search = 60 # Número de nuevas configuraciones para los tk_ck que se prueban. (SE REDUCE A MEDIDA QUE CRECE LE NUM DE IT GLOBAL)
    Range_Ntkck=range(N_tk_ck_search)
    width_tk_search= 7 #genera aleatoriamente entre tk-width y tk+width
    min_d = 21  # tiempo mínimo entre tratamientos
    depth_ck_search = 10  # Número de puntos de cría que cambian de valor en la exploración respecto al mejor hasta ahora

    #Method parameters
    Method = 'Radau'
    r_tol = 1e-4   #for C=3240, 1e-9, C=2430, 1e-6, C=1620, e-5, C=810, e-4
    cst_L = 0.25
    cst_A = cst_L * Mu(t0) / (2 * DeltaA(t0))


    # # # # # #RESULTS
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    filename_tkck = f'All_tkckJtY_{current_time}_{Year}_{C}.pkl'
    filename_J_tracker = f'All_J_tracker_{current_time}_{Year}_{C}.pkl'

    t1 = time.time()

    All_tkckJtY, All_J_tracker = simulation_tkck(C, t0, T,t_ini, t_fin, min_d, S, NJ, N_sims, N_iter,width_tk_search, depth_ck_search,filename_tkck,filename_J_tracker)

    with open(os.path.join(directory_path, 'Results', filename_tkck), 'wb') as f:
        pickle.dump(All_tkckJtY, f)

    with open(os.path.join(directory_path, 'Results', filename_J_tracker), 'wb') as f:
        pickle.dump(All_J_tracker, f)

    t2=time.time()

    print(t2-t1)


#
# # #
# r_tol=1e-8
# t1 = time.time()
# L0 = np.array([cst_L*K(t0)[i]*0.75 for i in Range_S])
# A0 = np.array([cst_A*K(t0)[i]*0.75 for i in Range_S])
# c0 = np.array([0 for i in Range_S])
# y0=np.concatenate((L0,A0,c0))
#
#
# # # tk=[ 94.14219363 ,133.51988809 ,163.61256478 ,256.49623993]
# ck = generate_ck(810, Gr_dose, S, NJ)
# #tk= [102, 168, 218, 270]
# tk=[t_ini+j*(t_fin-t_ini)/5 for j in range(1,5)]
# NJ=len(tk)
# Range_NJ = range(NJ)
# Range_NJ_1 = range(NJ - 1)
# Range_1_NJ = range(1, NJ)
#
#
#
# for x in [1e-7,1e-6,1e-5,1e-4]:
#     t1=time.time()
#     r_tol=x
#     t, Y = system_solve(y0, Gr_dose, tk, ck, t0, T)
#     J = crit(Y, t)
#     t2 = time.time()
#     print(J, t2-t1)

# with open(f'Results/{Year}/Controlled.pkl', 'wb') as f:
#     pickle.dump((J,t,Y), f)


#
#
# for i in range(20):
#     r_tol=1e-4
#     t1 = time.time()
#     L0 = np.array([cst_L*K(t0)[i]*(0.5+0.5*random.random()) for i in Range_S])
#     A0 = np.array([cst_A*K(t0)[i]*(0.5+0.5*random.random()) for i in Range_S])
#     c0 = np.array([0 for i in Range_S])
#     y0=np.concatenate((L0,A0,c0))
#
#
#     tk = generate_tk(t_ini, t_fin, min_d, NJ)
#     ck = generate_ck(810, Gr_dose, S, NJ)
#     #ck = np.array([[1 for i in Range_S] for j in Range_NJ])
#
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
# #
