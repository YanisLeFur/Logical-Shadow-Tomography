import numpy as np
import matplotlib.pyplot as plt
import cirq
from tqdm import tqdm
from scipy.linalg import fractional_matrix_power
from LST import *
from scipy.linalg import sqrtm


N =2
nsimu = 100
n_ave=1
N_U = 1000
circuit = base(N)
p_set= [0]#np.arange(0.,0.6,0.1)
mse = []
distance = []
distance2= []
distance3 = []
for proba in tqdm(p_set):
    dist=0.
    dist2=0.
    dist3=0.
    error= 0.
    for _ in range(n_ave):
        #rho_shadow = find_rho_clifford(N,1000,1,probability = proba)
        rho_shadow = find_rho(circuit,N,nsimu,probability=proba)
        #rho_shadow_NU = find_rho_N_U(circuit,N,N_U,nsimu,probability=proba)
        print("shadow",rho_shadow)
        #print("shadow_NU", rho_shadow_NU)
        rho_shadow2 = rho_shadow@rho_shadow.T.conj()
        rho_shadow3 = rho_shadow@rho_shadow2.T.conj()

        rho_shadow  /= np.trace(rho_shadow)
        rho_shadow2 /= np.trace(rho_shadow2)
        rho_shadow3 /= np.trace(rho_shadow3)


        simulator = cirq.Simulator()
        simulation = simulator.simulate(circuit)
        final_state = np.array([simulation.final_state_vector])
        rho_actual = final_state.T@final_state
        ngate = 1000
        gates= [np.random.choice(['X','Y','Z'],size=N) for _ in range(ngate)]
        observable,counts = np.unique(gates,axis = 0, return_counts =True)
        
        for obs in observable:
            T_shadow=np.trace(rho_shadow@np.kron(obs_gate(obs[0]),obs_gate(obs[1])))/np.trace(rho_shadow)
            T_actual = np.trace(rho_actual@np.kron(obs_gate(obs[0]),obs_gate(obs[1])))
            error+=(T_shadow-T_actual)**2

           
        
        dist += trace_dist(rho_actual,rho_shadow)
        dist2+= trace_dist(rho_actual,rho_shadow2)
        dist3 += trace_dist(rho_actual,rho_shadow3)
      
            
            
    mse.append((error/n_ave))
    distance.append(dist/n_ave)
    distance2.append(dist2/n_ave)
    distance3.append(dist3/n_ave)

print("trace distance",dist)


plt.subplot(121)
plt.suptitle("Correct")
plt.imshow(rho_actual.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_actual.imag,vmax=0.7,vmin=-0.7)
plt.show()
print("---")

plt.subplot(121)
plt.suptitle("Shadow(Full Clifford)")
plt.imshow(rho_shadow.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_shadow.imag,vmax=0.7,vmin=-0.7)
plt.show()


plt.figure()
plt.plot(p_set,mse,label= 'rho')
plt.grid()
plt.legend()
plt.xlabel("p")
plt.ylabel("MSE")



E = 2*N*p_set
plt.figure()
plt.plot(E,distance,label= 'rho')
plt.plot(E,distance2,label= 'rho2')
plt.plot(E,distance3,label= 'rho3')
plt.grid()
plt.legend()
plt.xlabel("p")
plt.ylabel("distance")



plt.figure()
plt.loglog(p_set,distance,label= 'rho')
plt.loglog(p_set,distance2,label= 'rho2')
plt.loglog(p_set,distance3,label= 'rho3')
plt.grid()
plt.legend()
plt.xlabel("p")
plt.ylabel("distance")
#plt.show()




