import numpy as np
import cirq
import jax.numpy as jnp
from multiprocessing import Pool
from tqdm import tqdm


def int_to_binary(N,b):
    bit = format(b,'b')
    if len(bit)<N:
        for _ in range(N-len(bit)):bit = str(0)+bit
    return bit

def base(qubits):
    return cirq.Circuit(
            cirq.H(qubits[0]),
            *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, len(qubits))],
            )

def measure(c,qubits):
    c.append(cirq.measure(*qubits, key='result'))
    return c


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
    if g=="X":
        return cirq.unitary(cirq.X)
    elif g=="Y":
        return cirq.unitary(cirq.Y)
    elif g=="Z":
        return cirq.unitary(cirq.Z)
    elif g=="I":
        return cirq.unitary(cirq.I)
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

def operator(string):
    op = 1.
    for o in string:
        op = np.kron(obs_gate(o),op)
    return op



def reconstruct(packed_arg):
    labels,results = packed_arg
    N=10
    shadows = jnp.zeros((2**N,2**N),dtype=jnp.complex64)
    shots = 0
    I = np.eye(2)
    for pauli_string,counts in zip(labels,tqdm(results)):
        # iterate over measurements
        for bit,count in counts.items():
            bit =  int_to_binary(N,bit)
            mat = 1.
            for i,bi in enumerate(bit):
                mat = jnp.kron(mat , 3 * rotGate(pauli_string[i],int(bi)) - I )    
            shadows+=mat*count
            shots+=count
    return shadows/shots

def create_mult_batch(array,num_batch):
    return  np.array_split(array,num_batch)



if __name__ == '__main__':
    nsimu =6
    labels = np.load("labels_QVM.npy",allow_pickle=True)
    results = np.load("results_QVM.npy",allow_pickle=True)

    label_batch = create_mult_batch(labels,nsimu)
    results_batch = create_mult_batch(results,nsimu)

    packed_arg = [(label_batch[i],results_batch[i])for i in range(nsimu)]
    p = Pool(nsimu)
    results=tqdm(p.map(reconstruct,packed_arg))
    p.close()
    rho_shadow =[r for r in results]
    np.save("rho_shadow_nsimu=4e7_QVM",np.sum(rho_shadow,axis = 0))
