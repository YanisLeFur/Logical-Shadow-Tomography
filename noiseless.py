import numpy as np
import matplotlib.pyplot as plt
import cirq
from scipy.linalg import sqrtm
from numpy.linalg import matrix_power


def int_to_binary(N,b):
    bit = format(b,'b')
    if len(bit)<N:
        for _ in range(N-len(bit)):bit = str(0)+bit
    return bit

def base(N,c):
    '''create a N qubit GHZ state'''
    c.append(cirq.H(q[0]))
    if N>=2: c.append(cirq.CNOT(q[0], q[1]))
    if N>=3: c.append(cirq.CNOT(q[0], q[2]))
    if N>=4: c.append(cirq.CNOT(q[1], q[3]))
    return c

def measure(c):
    if N==2: c.append(cirq.measure(q[1],q[0],key = "result"))
    if N==3: c.append(cirq.measure(q[2],q[1],q[0],key = "result"))
    if N==4: c.append(cirq.measure(q[3],q[2],q[1],q[0],key = "result"))

    return c

def bitGateMap(c,g,qi):
    '''Map X/Y/Z string to cirq ops'''
    if g=="X":
        c.append(cirq.H(q[qi]))
        
    elif g=="Y":
        sdg = cirq.S**-1
        c.append(sdg(q[qi]))
        c.append(cirq.H(q[qi]))
        
    elif g=="Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")
def Minv(N,X):
    '''inverse shadow channel'''
    return ((2**N+1.))*X - np.eye(2**N)

def rotGate(g):
    '''produces gate U such that U|psi> is in Pauli basis g'''
    if g=="X":
        return 1/np.sqrt(2)*np.array([[1.,1.],[1.,-1.]])
    elif g=="Y":
        return 1/np.sqrt(2)*np.array([[1.,-1.0j],[1.,1.j]])
    elif g=="Z":
        return np.eye(2)
    else:
        raise NotImplementedError(f"Unknown gate {g}")    
    

def obs_gate(g):
    if g=="X":
        return cirq.unitary(cirq.X)
    elif g=="Y":
        return cirq.unitary(cirq.Y)
    elif g=="Z":
        return cirq.unitary(cirq.Z)
    else:
        raise NotImplementedError(f"Unknown gate {g}")    

def trace_dist(lam_exact,rho):
    ''' returns normalized trace distance between lam_exact and rho'''
    mid = (lam_exact-rho).conj().T@(lam_exact-rho)
    N = 2**int(np.log2(lam_exact.shape[0])/2)
    # svd mid and apply sqrt to singular values
    # based on qiskit internals function
    U1,d,U2 = np.linalg.svd(mid)
    sqrt_mid = U1@np.diag(np.sqrt(d))@U2
    dist = np.trace(sqrt_mid)/2
    return dist/N




N =2
nsimu = 10000
dist=[]

for N in range(2,3):
    results ={}
    gates= [np.random.choice(['X','Y','Z'],size=N) for _ in range(nsimu)]
    labels,counts = np.unique(gates,axis = 0, return_counts =True)
    print(labels)

    circuit = cirq.Circuit()
    q = cirq.LineQubit.range(N)
    base(N,circuit)
    'Generate the measurements outcomes'
    results = []
    for bit_string,count in zip(labels,counts):
        c_m = circuit.copy()
        #rotate the basis for each qubits
        for i,bit in enumerate(bit_string): bitGateMap(c_m,bit,i) 
        measure(c_m)
        s = cirq.Simulator()
        samples = s.run(c_m, repetitions=count)
        counts = samples.histogram(key='result')
        results.append(counts)



    'Shadow Tomography part'
    shadows = []
    shots = 0
    for pauli_string,counts in zip(labels,results):
        # iterate over measurements
        for bit,count in counts.items():
            bit = int_to_binary(N,bit)
            mat = 1.
            for i,bi in enumerate(bit[::-1]):
                b = rotGate(pauli_string[i])[int(bi),:]
                mat = np.kron(Minv(1,np.outer(b.conj(),b)),mat)
            shadows.append(mat*count)
            shots+=count
    print(shadows)
    rho_shadow = np.sum(shadows,axis=0)/(shots)



    simulator = cirq.Simulator()
    simulation = simulator.simulate(circuit)
    final_state = np.array([simulation.final_state_vector])
    rho_actual = final_state.T@final_state
    dist.append(trace_dist(rho_actual,rho_shadow))
    dist.append(trace_dist(rho_actual,rho_shadow@rho_shadow))
    dist.append(trace_dist(rho_actual,rho_shadow@rho_shadow@rho_shadow))



plt.subplot(121)
plt.suptitle("Correct")
plt.imshow(rho_actual.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_actual.imag,vmax=0.7,vmin=-0.7)

plt.figure()
plt.subplot(121)
plt.suptitle("Shadow(Pauli)")
plt.imshow(rho_shadow.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_shadow.imag,vmax=0.7,vmin=-0.7)
print("dist",dist)
'''
n=[2,3,4]
plt.figure()
plt.loglog(n,dist)
plt.grid()
plt.show()
'''