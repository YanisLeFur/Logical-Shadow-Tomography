
from multiprocessing import Pool
import cirq
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import qsimcirq
import jax.numpy as jnp

def str_to_vec(b):
    vec = np.array([0,0])


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
    


#Encoding 2 logical qubits in the [5,1,3] code

def encode_2_logical(state1,state2,qubits):
    CZ =  cirq.ControlledGate(sub_gate=cirq.Z, num_controls=1)
    CY = cirq.ControlledGate(sub_gate=cirq.Y,num_controls=1)
    circuit = cirq.Circuit()
    

    #first logical qubit
    if state1 == "1":
        circuit.append(cirq.X(qubits[4]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.H(qubits[2]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.CNOT(qubits[0],qubits[4]))
    circuit.append(cirq.CNOT(qubits[1],qubits[4]))
    circuit.append(cirq.CNOT(qubits[2],qubits[4]))
    circuit.append(cirq.CNOT(qubits[3],qubits[4]))
    circuit.append(CZ(qubits[0],qubits[1]))
    circuit.append(CZ(qubits[1],qubits[2]))
    circuit.append(CZ(qubits[2],qubits[3]))
    circuit.append(CZ(qubits[3],qubits[4]))
    circuit.append(CZ(qubits[0],qubits[4]))

    #second logical qubit
    if state2 == "1":
        circuit.append(cirq.X(qubits[9]))
    circuit.append(cirq.H(qubits[5]))
    circuit.append(cirq.H(qubits[6]))
    circuit.append(cirq.H(qubits[7]))
    circuit.append(cirq.H(qubits[8]))
    circuit.append(cirq.CNOT(qubits[5],qubits[9]))
    circuit.append(cirq.CNOT(qubits[6],qubits[9]))
    circuit.append(cirq.CNOT(qubits[7],qubits[9]))
    circuit.append(cirq.CNOT(qubits[8],qubits[9]))
    circuit.append(CZ(qubits[5],qubits[6]))
    circuit.append(CZ(qubits[6],qubits[7]))
    circuit.append(CZ(qubits[7],qubits[8]))
    circuit.append(CZ(qubits[8],qubits[9]))
    circuit.append(CZ(qubits[5],qubits[9]))
    return circuit


#define the logical HADAMARD gate from Fig.2 of https://arxiv.org/abs/1603.03948
def Hadamard_logical(circuit,qubits):
    for i in range(5):circuit.append(cirq.H(qubits[i]))
    circuit.append(cirq.SWAP(qubits[0],qubits[1]))
    circuit.append(cirq.SWAP(qubits[1],qubits[4]))
    circuit.append(cirq.SWAP(qubits[4],qubits[3]))
    pass

#define the logical non transversal CNOT gate from Fig.15 of https://arxiv.org/abs/2208.01863
def CNOT_logical(circuit,qubits):
    sdg = cirq.S**-1

    #local clifford rotation (1st qubit)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.S(qubits[0]))
    circuit.append(cirq.Y(qubits[2]))
    circuit.append(cirq.H(qubits[4]))
    circuit.append(cirq.S(qubits[4]))
    #local clifford rotation (2nd qubit)
    circuit.append(cirq.H(qubits[5]))
    circuit.append(cirq.S(qubits[5]))
    circuit.append(cirq.Y(qubits[7]))
    circuit.append(cirq.H(qubits[9]))
    circuit.append(cirq.S(qubits[9]))
    
    #logical gate
    circuit.append(cirq.CNOT(qubits[0],qubits[5]))
    circuit.append(cirq.CNOT(qubits[2],qubits[7]))
    circuit.append(cirq.CNOT(qubits[4],qubits[9]))
    circuit.append(cirq.CNOT(qubits[0],qubits[9]))
    circuit.append(cirq.CNOT(qubits[2],qubits[5]))
    circuit.append(cirq.CNOT(qubits[4],qubits[7]))
    circuit.append(cirq.CNOT(qubits[0],qubits[7]))
    circuit.append(cirq.CNOT(qubits[2],qubits[9]))
    circuit.append(cirq.CNOT(qubits[4],qubits[5]))
    
    #local clifford rotation (1st qubit)
    circuit.append(sdg(qubits[0]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.Y(qubits[2]))
    circuit.append(sdg(qubits[4]))
    circuit.append(cirq.H(qubits[4]))
    #local clifford rotation (2nd qubit)
    circuit.append(sdg(qubits[5]))
    circuit.append(cirq.H(qubits[5]))
    circuit.append(cirq.Y(qubits[7]))
    circuit.append(sdg(qubits[9]))
    circuit.append(cirq.H(qubits[9]))
    pass

def GHZ_preparation(circuit,qubits):
    Hadamard_logical(circuit,qubits)
    CNOT_logical(circuit,qubits)
    
    return circuit

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
    

def Minv(N,X):
    '''inverse shadow channel'''
    return ((2**N+1.))*X - np.eye(2**N)

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


def generate_measurements(packed_arg):
    probability = 0.02
    labels,counts = packed_arg

    qubits = cirq.LineQubit.range(10)
    circuit  = encode_2_logical("0","0",qubits)
    GHZ_preparation(circuit,qubits)
    'Generate the measurements outcomes'
    results = []
    for bit_string,count in zip(labels,tqdm(counts)):
        c_m = circuit.copy()
        #rotate the basis for each qubits
        for i,bit in enumerate(bit_string):  bitGateMap(c_m,bit,i,qubits)
        measure(c_m,qubits)
        if probability>0.:
            c_m = c_m.with_noise(cirq.depolarize(p=probability))
        s = qsimcirq.QSimSimulator()
        samples = s.run(c_m, repetitions=count)
        counts = samples.histogram(key='result')
        results.append(counts)
    return results

def create_mult_batch(array,num_batch):
    return  np.array_split(array,num_batch)



if __name__ == '__main__':
    nsimu =6
    nsimu_scheme = 40000000
    p = Pool(nsimu)
    N=10
    scheme = [np.random.choice(['X','Y','Z'],size=N) for _ in tqdm(range(nsimu_scheme))]
    labels, counts = np.unique(scheme,axis=0,return_counts=True)

    label_batch = create_mult_batch(labels,nsimu)
    counts_batch = create_mult_batch(counts,nsimu)
    packed_arg = [(label_batch[i],counts_batch[i])for i in range(nsimu)]

    r_results=p.map(generate_measurements,packed_arg)
    p.close()
    results = [r for r in r_results]
    conc_results = np.concatenate((results[0],results[1]))
    for i in range(2,len(results)):conc_results = np.concatenate((conc_results,results[i]))

    print(len(conc_results))
    np.save("results_test",conc_results)
    np.save("labels_test",labels)
