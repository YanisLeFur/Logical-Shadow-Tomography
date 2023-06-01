import numpy as np
from LST import *
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
import itertools



N = 5
N_U = np.logspace(2.5,4,5)
N_S =np.logspace(0,4,5)
observables =["ZZIII","IZZII","IIZZI","IIIZZ","XXXXX"]
n_ave = 4
p  = 0.1
circuit = base(N)

#function for multiprocessing
def f(params):
    nu = params[0]
    nsimu= params[1]
    N=5
    p=0.1
    circuit = base(N)
    return  find_rho_N_U(circuit,N,int(nu),int(nsimu),probability=p)


'''
if __name__ == '__main__':
    #Real State
    simulator = cirq.Simulator()
    simulation = simulator.simulate(circuit)
    final_state = np.array([simulation.final_state_vector])
    rho_actual = final_state.T@final_state
    T_actual = []
    for obs in observables:
        O = 1.
        for o in obs:
            O = np.kron(obs_gate(o),O)
        T_actual.append(np.trace(rho_actual@O))


    paramlist = list(itertools.product(N_U,N_S)) 
    
    shadows =[] 
    pool = multiprocessing.Pool(1)
    shadows = list(tqdm(pool.imap(f,paramlist),total=len(paramlist)))   #pool.map(f,tqdm(paramlist))
    print(shadows)
    delta = []
    for s in shadows:
        T_shadow = []
        for obs in observables:
            O = 1.
            for o in obs:
                O = np.kron(obs_gate(o),O)
            T_shadow.append(np.trace(s@O))
        delta.append(np.log10((np.sum([T_actual[i]-T_shadow[i] for i in range(len(observables))],axis =0))**2))
    print(delta)
    np.savetxt('delta.out', delta)

    font = {'family' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)


    cs = plt.contourf(N_S, N_U, np.array(delta).reshape(len(N_U),len(N_S)), 
                  cmap ='plasma',
                  extend ='both',
                  alpha = 1)
  
    cbar = plt.colorbar(cs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_S$')
    plt.ylabel(r'$N_U$')
    cbar.set_label(r'$\log_{10}(\Delta_{GHZ}^2)$')
  
    plt.show()
'''
#Real State
simulator = cirq.Simulator()
simulation = simulator.simulate(circuit)
final_state = np.array([simulation.final_state_vector])
rho_actual = final_state.T@final_state
np.savetxt("rho_actual.txt",rho_actual)
T_actual = []
for obs in observables:
    O = 1.
    for o in obs:
        O = np.kron(obs_gate(o),O)
    T_actual.append(np.trace(rho_actual@O))
    
#Shadow Tomography
delta = []
for nsimu in tqdm(N_S):
    for nu in tqdm(N_U):
        delta_temp = []
        for _ in range(n_ave):
            rho_shadow = find_rho_N_U(circuit,N,int(nu),int(nsimu),probability=p)
            T_shadow = []
            for obs in observables:
                O = 1.
                for o in obs:
                    O = np.kron(obs_gate(o),O)
                T_shadow.append(np.trace(rho_shadow@O)/np.trace(rho_shadow))
            delta_temp.append(np.log10((np.sum([(T_actual[i]-T_shadow[i])**2 for i in range(len(observables))],axis =0))))
        delta.append(np.mean(delta_temp))
        np.savetxt('delta.out', delta)
print(delta)


font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


cs = plt.contourf(N_S, N_U, np.array(delta).reshape(len(N_U),len(N_S)), 
                  cmap ='plasma',
                  extend ='both',
                  alpha = 1)
  
cbar = plt.colorbar(cs)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N_S$')
plt.ylabel(r'$N_U$')
cbar.set_label(r'$\log_{10}(\Delta_{GHZ}^2)$')
  
plt.show()

