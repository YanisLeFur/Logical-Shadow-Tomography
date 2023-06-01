import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from LST import *

font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)



N = 5
N_U = 1500
N_S = 500
nsimu = 10000
n_average = 4
circuit =  base(N)
print(circuit)
p = 0.1
observables = ["ZZIII","IZZII","IIZZI","IIIZZ","XXXXX"]
delta = {}
for obs in observables:
     for i in range(3):
          delta[obs+str(i)]=[]

simulator = cirq.Simulator()
simulation = simulator.simulate(circuit)
final_state = np.array([simulation.final_state_vector])
rho_actual = final_state.T@final_state
np.savetxt("rho_actual.txt",rho_actual)

for _ in range(n_average):
    #rho_shadow = find_rho_clifford(N,N_U,N_S,probability=p)
    #rho_shadow = find_rho(circuit,N,nsimu,probability=p)
    rho_shadow =  find_rho_N_U(circuit,N,int(N_U),int(N_S),probability=p)
    np.savetxt("rho_shadow.txt",rho_shadow)
    rho_shadow2 = rho_shadow@rho_shadow
    rho_shadow3 = rho_shadow@rho_shadow2
    rho_shadow/=np.trace(rho_shadow)
    rho_shadow2/=np.trace(rho_shadow2)
    rho_shadow3/=np.trace(rho_shadow3)
    print("trace dist rho",trace_dist(rho_shadow,rho_actual))
    print("trace dist rho2",trace_dist(rho_shadow2,rho_actual))
    print("trace dist rho3",trace_dist(rho_shadow3,rho_actual))
    for obs in observables:
        O = 1.
        for o in obs:

            O = np.kron(obs_gate(o),O)
        print("obs rho:",np.trace(rho_actual@O))
        print("obs rho:",np.trace(rho_shadow@O)/np.trace(rho_shadow))
        print("obs rho2:",np.trace(rho_shadow2@O)/np.trace(rho_shadow2))
        print("obs rho3:",np.trace(rho_shadow3@O)/np.trace(rho_shadow3))
        delta[obs+str(0)].append(np.trace(rho_shadow@O)/np.trace(rho_shadow))
        delta[obs+str(1)].append(np.trace(rho_shadow2@O)/np.trace(rho_shadow2))
        delta[obs+str(2)].append(np.trace(rho_shadow3@O)/np.trace(rho_shadow3))

print(delta)
with open('delta.txt', 'w') as f:
    print(delta, file=f)

x1 = []
x2 = []
x3 = []
err_x1 = []
err_x2 = []
err_x3 = []
for obs in observables:
    for i in range(3):
        print("Mean"+obs+str(i)+": ",np.mean(delta[obs+str(i)]),"var"+obs+str(i)+": ",np.var(delta[obs+str(i)]))
    x1.append(np.mean(delta[obs+str(0)]))
    x2.append(np.mean(delta[obs+str(1)]))
    x3.append(np.mean(delta[obs+str(2)]))
    err_x1.append(np.var(delta[obs+str(0)]))
    err_x2.append(np.var(delta[obs+str(1)]))
    err_x3.append(np.var(delta[obs+str(2)]))

etiquette = [r'$Z_1Z_2$',r'$Z_2Z_3$',r'$Z_3Z_4$',r'$Z_4Z_5$',r'$\prod_i X_i$']

# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(etiquette))
# Largeur des barres
largeur = .25

# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots()
r1 = ax.bar(position - largeur, x1,yerr = err_x1,width =  largeur,color = "grey",label = r'$\rho$')
r2 = ax.bar(position , x2, yerr = err_x2,width = largeur,color = "blue",label = r'$\rho^2$')
r3 = ax.bar(position + largeur, x3,yerr = err_x3,width = largeur,color = "purple",label = r'$\rho^3$')

# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(etiquette)
plt.legend()
plt.show()

