import numpy as np
import rustworkx as rx
from collections import Counter
from qiskit_ibm_runtime import RuntimeDecoder

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])

def zig_zag(qubits_in_row, layout, top=True, x_offset=0, y_offset=0, scale=1,roll=0):
    n_in_row = len(qubits_in_row)
    coords = np.array([0,scale/2,scale,scale/2])
    if top == True:
        shifts = (coords.tolist()*(n_in_row))[roll:roll+len(qubits_in_row)]
    else:
        shifts = ((-coords).tolist()*(n_in_row))[roll:roll+len(qubits_in_row)]
    coords = np.roll(coords, -roll)
    for i,(q,s) in enumerate(zip(qubits_in_row, shifts)):
        layout[q] = [i-x_offset, s-y_offset]

def inbetween(qubit_in_row, layout, scale=4, x_offset=0, y_offset=0):
    for ind, q in enumerate(qubit_in_row):
        layout[q] = [ind*scale + x_offset, -y_offset]

def eagle_layout():
    layout = {}
    zig_zag([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], layout, top=True, y_offset=0)
    zig_zag([18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], layout, top=False, y_offset=3)
    zig_zag([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51], layout, top=True, y_offset=8)
    zig_zag([56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70], layout, top=False, y_offset=11)
    zig_zag([75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], layout, top=True, y_offset=16)
    zig_zag([94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108], layout, top=False, y_offset=19)
    zig_zag([113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126], layout, top=True, y_offset=24, roll=1, x_offset=-1)
    inbetween([14, 15, 16, 17], layout, y_offset=1.5)
    inbetween([33, 34, 35, 36], layout, y_offset=5.5, x_offset=2)
    inbetween([52, 53, 54, 55], layout, y_offset=9.5)
    inbetween([71, 72, 73, 74], layout, y_offset=13.5, x_offset=2)
    inbetween([90, 91, 92, 93], layout, y_offset=17.5)
    inbetween([109, 110, 111, 112], layout, y_offset=21.5, x_offset=2)
    return layout

def falcon_layout():
    layout = {}
    zig_zag([0, 1, 4, 7, 10, 12, 15, 18, 21, 23], layout, top=True, y_offset=0, roll=3)
    zig_zag([3, 5, 8, 11, 14, 16, 19, 22, 25, 26], layout, top=False, y_offset=2, x_offset=-1)
    inbetween([6,17], layout, y_offset=-2,x_offset=3)
    inbetween([2,13,24], layout, y_offset=1, x_offset=1)
    inbetween([9,20], layout, y_offset=4, x_offset=3)
    return layout

def get_active_qubits(circuit):
    """
    """
    active_qubits = [[circuit.find_bit(q).index for q in step.qubits] for step in circuit.data]
    active_qubits = [a for b in active_qubits for a in b]
    return set(active_qubits)

def graph_from_circuit(circuit, n_qubits=None):
    """
    """
    if n_qubits is None:
        n_qubits = circuit.num_qubits
    edges = [
        tuple([circuit.find_bit(q).index for q in step[1]])
        for step in circuit.data if step[0].name!='barrier' and len(step[1])>1
    ]
    weighted_edges = [(u,v,w) for (u,v),w in Counter(edges).items()]
    circuit_coupling_graph = rx.PyGraph()
    circuit_coupling_graph.add_nodes_from(range(n_qubits))
    circuit_coupling_graph.add_edges_from(weighted_edges)
    return circuit_coupling_graph

def join_registers(count_dict):
    """
    """
    bitstr, freq = zip(*count_dict.items())
    return dict(zip([b.replace(' ', '') for b in bitstr], freq))

def _split_registers(string, sizes):
    """
    """
    parts = []
    prev = 0
    for s in sizes:
        parts.append(string[prev:prev+s])
        prev+=s
    return ' '.join(parts)

def split_registers(count_dict, registers):
    """
    """
    sizes = [reg.size for reg in registers]
    return  {
        _split_registers(bitstr, sizes):val for bitstr,val in count_dict.items()
    }

def unitary_2x2(a,b,theta,phi):
    return np.exp(1j * phi / 2) * np.array(
        [
            [np.exp(1j*a)*np.cos(theta), np.exp(1j*b)*np.sin(theta)],
            [-np.exp(-1j*b)*np.sin(theta), np.exp(-1j*a)*np.cos(theta)]
        ]
    )

def get_neighbourhood(circuit, seed, distance, neighbourhood=None):
    """ Recursively expand the neighbourhood to include connected qubits around the seed
    """

    if neighbourhood is None:
        neighbourhood = {seed}

    if distance == 0:
        return sorted(neighbourhood.difference({seed})) # remove original seeded qubit

    expanded_neighbourhood = neighbourhood.copy()
    for n in neighbourhood:
        for op in circuit:
            op_qubits = [q.index for q in op.qubits]
            if n in op_qubits:
                expanded_neighbourhood = expanded_neighbourhood.union(op_qubits)
    
    if len(expanded_neighbourhood) == len(neighbourhood):
        return sorted(neighbourhood.difference({seed}))
    
    return get_neighbourhood(
        circuit  = circuit, 
        seed     = seed, 
        distance = distance - 1, 
        neighbourhood = expanded_neighbourhood
    )

def get_expectation_values(_01_vals):
    """
    """
    if _01_vals.ndim == 2:
        _0 = _01_vals[:,0]
        _1 = _01_vals[:,1]
    elif _01_vals.ndim == 3:
        _0 = _01_vals[:,:,0]
        _1 = _01_vals[:,:,1]
    else:
        raise ValueError(f'Incorrect number of dimensions {_01_vals.ndim}')
    p0   = _0 + _1
    vals = _0 - _1
    vals = vals / p0
    return vals, p0

class FinishedJob:
    """ Intended to replicate the RuntimeJob: only status and result currently implemented
    """
    def __init__(self, filename):
        with open(filename, "r") as text_file:
            datastring = text_file.read()
        decoder = RuntimeDecoder()
        split_data = datastring.split('*_*_*')
        self.id = split_data[0]
        (
            self.inputs,
            self.experiments,
            self.metadata, 
            self.num_experiments, 
            self.quasi_dists
        ) = map(decoder.decode, split_data[1:])

    def job_id(self):
        return self.id
    
    def status(self):
        class blank:
            pass
        out = blank()
        out.name = "DONE"
        return out
    
    def result(self):
        class blank:
            pass
        out = blank()
        out.experiments = self.experiments
        out.metadata = self.metadata
        out.num_experiments = self.num_experiments
        out.quasi_dists = self.quasi_dists
        return out