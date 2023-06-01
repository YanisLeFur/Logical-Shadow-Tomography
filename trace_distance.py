import numpy as np
import matplotlib.pyplot as plt
import cirq
from tqdm import tqdm
from scipy.linalg import fractional_matrix_power
from LST import *
from scipy.linalg import sqrtm
from numpy.linalg import matrix_power
import matplotlib

font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


N =2
nsimu = 10000
n_ave=5
N_U = 1500
circuit = base(N)
p_set= np.arange(0.,0.6,0.01)
distance = []
distance2= []
distance3 = []
distance10 = []
for proba in tqdm(p_set):
    dist=0.
    dist2=0.
    dist3=0.
    dist10=0.
    error= 0.
    for _ in range(n_ave):
        rho_shadow = find_rho(circuit,N,nsimu,probability=proba)
        #np.savetxt('rho_shadow_N='+str(N)+"_N_S="+str(nsimu),rho_shadow)
        #rho_shadowNU = find_rho_N_U(circuit,N,N_U,50,probability=proba)
        #np.savetxt('rho_shadow_N='+str(N)+"_N_U="+str(N_U)+"_N_S="+str(nsimu),rho_shadow)
        rho_shadow2 = rho_shadow@rho_shadow.T.conj()
        rho_shadow3 = rho_shadow@rho_shadow2.T.conj()
        rho_shadow10 = matrix_power(rho_shadow,10)

        rho_shadow  /= np.trace(rho_shadow)
        rho_shadow2 /= np.trace(rho_shadow2)
        rho_shadow3 /= np.trace(rho_shadow3)
        rho_shadow10 /= np.trace(rho_shadow10)

        simulator = cirq.Simulator()
        simulation = simulator.simulate(circuit)
        final_state = np.array([simulation.final_state_vector])
        rho_actual = final_state.T@final_state
        #print(rho_actual)
               
        dist += trace_dist(rho_actual,rho_shadow)
        dist2+= trace_dist(rho_actual,rho_shadow2)
        dist3 += trace_dist(rho_actual,rho_shadow3)
        dist10 += trace_dist(rho_actual,rho_shadow10)
        
    distance.append(dist/n_ave)
    distance2.append(dist2/n_ave)
    distance3.append(dist3/n_ave)
    distance10.append(dist10/n_ave)
    print(distance,distance2,distance3,distance10)

print("dist",distance,distance2,distance3)



E = 2*N*p_set
plt.figure()
plt.plot(E,distance,label= r'$\rho$')
plt.plot(E,distance2,label= r'$\rho^2$')
plt.plot(E,distance3,label= r'$\rho^3$')
plt.plot(E,distance10,label= r'$\rho^{10}$')
plt.grid()
plt.legend()
plt.xlabel("Expected number of error")
plt.ylabel("trace distance")



plt.figure()
plt.loglog(E,distance,label= r'$\rho$')
plt.loglog(E,distance2,label= r'$\rho^2$')
plt.loglog(E,distance3,label= r'$\rho^3$')
plt.loglog(E,distance10,label= r'$\rho^{10}$')
plt.grid()
plt.legend()
plt.xlabel("Expected number of error")
plt.ylabel("Trace distance to ideal state")
plt.show()
