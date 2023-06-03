import cirq
import cirq_google
import qsimcirq
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp


def base(qubits):
   return cirq.Circuit(
            cirq.H(qubits[0]),
            *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, len(qubits))],
            )

def measure(circuit,qubits):
   circuit.append(cirq.measure(*qubits, key='result'))
   return circuit

def int_to_binary(N,b):
    bit = format(b,'b')
    if len(bit)<N:
        for _ in range(N-len(bit)):bit = str(0)+bit
    return bit

def string_to_op(ope):
    if ope == 'X': return cirq.unitary(cirq.X)
    if ope == 'Y': return cirq.unitary(cirq.Y)
    if ope == 'Z': return cirq.unitary(cirq.Z)

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



def find_rho_N_U(circuit,qubits,device_qubit_chain,processor_id,gate_type,N=2,N_U=100,N_S = 10000):

    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    noise_props = cirq_google.noise_properties_from_calibration(cal)
    noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    sim = qsimcirq.QSimSimulator(noise=noise_model)
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(
                processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal}
                )
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])


    gates= [np.random.choice(['X','Y','Z'],size=N) for _ in range(N_U)]
    results = []
    for bit_string in tqdm(gates):
        c_m = circuit.copy()
        for i,bit in enumerate(bit_string): bitGateMap(c_m,bit,i,qubits) 
        measure(c_m,qubits) 
    
        translated_ghz_circuit = cirq.optimize_for_target_gateset(
                                c_m, context=cirq.TransformerContext(deep=True), gateset=gate_type
                                )   
        qubit_map = dict(zip(qubits, device_qubit_chain))
        device_ready_ghz_circuit = translated_ghz_circuit.transform_qubits(lambda q: qubit_map[q])
        samples = sim_engine.get_sampler(processor_id).run(device_ready_ghz_circuit,repetitions=N_S)
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





