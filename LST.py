import numpy as np
import cirq
from tqdm import tqdm
import qsimcirq
import os
import jax.numpy as jnp

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def int_to_binary(N,b):
    bit = format(b,'b')
    if len(bit)<N:
        for _ in range(N-len(bit)):bit = str(0)+bit
    return bit

def string_to_op(ope):
    if ope == 'X': return cirq.unitary(cirq.X)
    if ope == 'Y': return cirq.unitary(cirq.Y)
    if ope == 'Z': return cirq.unitary(cirq.Z)



def base(qubits):
   return cirq.Circuit(
            cirq.H(qubits[0]),
            *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, len(qubits))],
            )

def measure(circuit,qubits):
   circuit.append(cirq.measure(*qubits, key='result'))
   return circuit

def str_to_vec(b):
    vec = np.array([0,0])

    vec[int(b)]=1
    return vec


def bitGateMap(c,g,qi,qubits):
    '''Map X/Y/Z string to cirq ops'''
    if g=="X":
        c.append(cirq.H(qubits[qi]))
        
    elif g=="Y":
        sdg = cirq.S**-1
        c.append(sdg(qubits[qi]))
        c.append(cirq.H(qubits[qi]))
        
    elif g=="Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")



def rotGate(g,bi):
    '''produces gate U such that U|psi> is in Pauli basis g'''
    if g=="X":
        if bi==0:
            return jnp.array([[0.5,0.5],[0.5,0.5]])
        else:
            return jnp.array([[0.5,-0.5],[-0.5,0.5]])

    elif g=="Y":
        if bi==0:
            return jnp.array([[0.5,-0.5j],[0.5j,0.5]])
        else:
            return jnp.array([[0.5,0.5j],[-0.5j,0.5]])
    elif g=="Z":
        if bi==0:
            return jnp.array([[1.,0.],[0.,0.]])
        else:
            return jnp.array([[0.,0.],[0.,1.]])
    else:
        raise NotImplementedError(f"Unknown gate {g}")
    

def obs_gate(g):
    if g == "I":
        return cirq.unitary(cirq.I)
    if g=="X":
        return cirq.unitary(cirq.X)
    elif g=="Y":
        return cirq.unitary(cirq.Y)
    elif g=="Z":
        return cirq.unitary(cirq.Z)
    else:
        raise NotImplementedError(f"Unknown gate {g}")    

def find_rho(circuit,qubits,N=2,nsimu = 10000,probability=0):
    gates= [np.random.choice(['X','Y','Z'],size=N) for _ in range(nsimu)]
    labels,counts = np.unique(gates,axis = 0, return_counts =True)
    results = []
    for bit_string,count in tqdm(zip(labels,counts)):
        c_m = circuit.copy()
        for i,bit in enumerate(bit_string): bitGateMap(c_m,bit,i,qubits) 
        measure(c_m,qubits) 
        if probability>0.:
            c_m = c_m.with_noise(cirq.depolarize(p=probability))
        s = qsimcirq.QSimSimulator()
        samples = s.run(c_m, repetitions=count)
        counts = samples.histogram(key='result')
        results.append(counts)

    shadows = []
    shots = 0
    I = np.eye(2)
    for pauli_string,counts in zip(gates,results):
        # iterate over measurements
        for bit,count in counts.items():
            bit = int_to_binary(N,bit)
            mat = 1.
            for i,bi in enumerate(bit):
                mat = jnp.kron(mat , 3 * rotGate(pauli_string[i],int(bi)) - I )
            shadows.append(mat*count)
            shots+=count
    return np.sum(shadows,axis=0)/(shots)

def find_rho_N_U(circuit,qubits,N=2,N_U=100,N_S = 10000,probability=0):

    gates= [np.random.choice(['X','Y','Z'],size=N) for _ in range(N_U)]
    results = []
    for bit_string in tqdm(gates):
        c_m = circuit.copy()
        for i,bit in enumerate(bit_string): bitGateMap(c_m,bit,i,qubits) 
        measure(c_m,qubits) 
        if probability>0.:
            c_m = c_m.with_noise(cirq.depolarize(p=probability))
        s = qsimcirq.QSimSimulator()
        samples = s.run(c_m, repetitions=N_S)
        counts = samples.histogram(key='result')
        results.append(counts)
    shadows = []
    shots = 0
    I = np.eye(2)
    for pauli_string,counts in zip(gates,results):
        for bit,count in counts.items():
            bit = int_to_binary(N,bit)
            mat = 1.
            for i,bi in enumerate(bit):
                mat = jnp.kron(mat , 3 * rotGate(pauli_string[i],int(bi)) - I )
            shadows.append(mat*count)
            shots+=count
    return np.sum(shadows,axis=0)/(shots)


def trace_dist(lam_exact,rho):
    ''' returns normalized trace distance between lam_exact and rho'''
    mid = (lam_exact-rho).conj().T@(lam_exact-rho)
    N = 2**int(np.log2(lam_exact.shape[0])/2)
    U1,d,U2 = np.linalg.svd(mid)
    sqrt_mid = U1@np.diag(np.sqrt(d))@U2
    dist = np.trace(sqrt_mid)/2
    return dist/N


