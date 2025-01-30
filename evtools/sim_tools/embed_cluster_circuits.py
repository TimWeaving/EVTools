import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from collections import Counter
from qiskit import transpile, QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qiskit.circuit import AncillaQubit, AncillaRegister, QuantumRegister
from qiskit.circuit.library import RXGate, RYGate, RZGate, SGate, SdgGate
from .utils import eagle_layout, falcon_layout, get_active_qubits, graph_from_circuit

class QubitClusters:
    """
    """

    def __init__(self, backend):
        """
        """
        self.backend = backend
        self.config  = backend.configuration()
        self.qpu_series = self.config.processor_type['family'].lower()
        if self.qpu_series == 'eagle':
            self.graph_layout = eagle_layout()
        elif self.qpu_series == 'falcon':
            self.graph_layout = falcon_layout()
        else:
            raise NotImplementedError('Currently only supports IBM Quantum Eagle and Falcon chips.')
        self.clusters   = {}
        self.n_clusters = 0
        
    def coupling_graph(self):
        """
        """
        coupling_graph = rx.PyGraph()
        coupling_graph.add_nodes_from(range(self.config.num_qubits))
        coupling_graph.add_edges_from([(a,b,None) for a,b in self.config.coupling_map])
        return coupling_graph
    
    def draw_topology(self):
        """
        """
        if self.qpu_series == 'eagle':
            figsize=(8,8)
        elif self.qpu_series == 'falcon':
            figsize=(8,5)
        fig,ax = plt.subplots(figsize=figsize)
        rx.visualization.mpl_draw(self.coupling_graph(), pos=self.graph_layout, with_labels=True, ax=ax,
            labels=lambda x:x, node_color='cyan', font_color='blue')
        return fig
    
    def get_cluster(self, qubits):
        """
        """
        return self.coupling_graph().subgraph(qubits)    

    def add_cluster(self, compute_qubits, ancilla_qubits=[]):
        """
        """
        new_clusters = self.clusters.copy()
        qubit_dict = {'full':sorted(compute_qubits+ancilla_qubits), 'compute':compute_qubits, 'ancilla':ancilla_qubits}
        new_clusters[self.n_clusters] = {'qubits':qubit_dict}
        cluster_union = [a for b in new_clusters.values() for a in b['qubits']['full']]
        assert len(cluster_union) == len(set(cluster_union)), 'Clusters and not disjoint - overlapping qubits'
        new_clusters[self.n_clusters]['graphs']   = dict(zip(qubit_dict.keys(), map(self.get_cluster, qubit_dict.values())))
        new_clusters[self.n_clusters]['n_qubits'] = dict(zip(qubit_dict.keys(), map(len, qubit_dict.values())))
        self.clusters = new_clusters
        self.n_clusters = len(new_clusters)

    def draw_cluster_graph(self, index, which='full'):
        """
        """
        fig,ax = plt.subplots(figsize=(5,5))
        rx.visualization.mpl_draw(self.clusters[index]['graphs'][which],with_labels=True,ax=ax,
            labels=lambda x:'$c_{'+str(x)+'}$', node_color='cyan', font_color='blue')
        return fig
    
    def total_clustered_qubits(self):
        return sum([v['n_qubits']['full'] for v in self.clusters.values()])
    
class EmbedClusterCircuits(QubitClusters):
    """
    """

    optimization_level = 3
    apply_cancellation = True

    def __init__(self, primitive_qc, backend):
        """
        """
        super().__init__(backend)
        self.primitive_qc = primitive_qc
        self.transpiled_circuits = {}
        self.echo_verification_circuits = {}
        self.ev_flag = False
        self.sys_qc  = None

        self.gate_cancellation = PassManager()
        primitive_params = [step.operation.params[0] for step in self.primitive_qc.data if (step.operation.params!=[] and not isinstance(step.operation.params[0], float))]
        for theta in primitive_params:
            self.gate_cancellation.append(InverseCancellation(gates_to_cancel=[(RXGate(theta), RXGate(-theta))]))
            self.gate_cancellation.append(InverseCancellation(gates_to_cancel=[(RYGate(theta), RYGate(-theta))]))
            self.gate_cancellation.append(InverseCancellation(gates_to_cancel=[(RZGate(theta), RZGate(-theta))]))
        self.gate_cancellation.append(InverseCancellation(gates_to_cancel=[(SGate(), SdgGate())]))
        self.gate_cancellation.append(CommutativeCancellation(basis_gates=['cx', 'x', 'y', 'z', 'h']))

    def reverse_qubit_order(self, q):
        """ Qiskit ordering is reversed
        """
        return self.primitive_qc.num_qubits - 1 - q
        
    def cancel_inverse_gates(self, circuit):
        """ Depending on the observable being measured, there may be many inverse 
        gate paits that can be cancelled in the echo verification circuit
        """
        temp_qc = circuit.copy()
        old_depth = 1
        new_depth = 0
        # need to run multiple passes since this will only cancel at most one CNOT pair each time
        # terminates once the PassManager finds no further gates that it can cancel
        while new_depth != old_depth:
            old_depth = len(temp_qc.data)
            temp_qc = self.gate_cancellation.run(temp_qc)
            new_depth = len(temp_qc.data)
        return temp_qc

    def prepare_echo_verification(self, paulistrings_to_measure, index):
        """
        """
        n_ancilla_qubits = self.clusters[index]['n_qubits']['ancilla']
        assert len(paulistrings_to_measure) <= n_ancilla_qubits, 'Insufficient number of ancilla to measure the specified paulistrings.'
        paulistring_lengths = list(set(map(len, paulistrings_to_measure)))
        assert len(paulistring_lengths) == 1, 'Pauli strings of differing qubit numbers'
        assert paulistring_lengths[0] == self.primitive_qc.num_qubits, 'Pauli string defined over a different number of qubits than the primitive circuit.'
        assert set([a for b in map(set, paulistrings_to_measure) for a in b]).issubset({'I', 'X', 'Y', 'Z'}), 'Pauli strings must be strings of I,X,Y,Z.'

        ev_circuit = QuantumCircuit()
        qr = QuantumRegister(self.primitive_qc.num_qubits, 'sys')
        ar = AncillaRegister(n_ancilla_qubits, 'anc')
        ev_circuit.add_register(qr)
        ev_circuit.add_register(ar)

        # copy primitive circuit to be converted in echo verification structure
        ev_circuit.compose(self.primitive_qc, qubits=qr, inplace=True)

        for ancilla_index, paulistring in enumerate(paulistrings_to_measure):
            non_identity_indices = [(q,P) for q,P in enumerate(paulistring) if P!='I']
            # change-of-basis
            for q,P in non_identity_indices:
                q = self.reverse_qubit_order(q)
                if P == 'Y':
                    ev_circuit.sdg(qr[q])
                if P != 'Z':
                    ev_circuit.h(qr[q])
            # store parity in ancilla qubit
            indices_to_couple,_ = zip(*non_identity_indices)
            indices_to_couple = list(map(self.reverse_qubit_order, indices_to_couple))
            qubit_pairs = list(zip(indices_to_couple[:-1], indices_to_couple[1:]))
            for q1,q2 in qubit_pairs:
                ev_circuit.cx(q1,q2)
            ev_circuit.cx(indices_to_couple[-1],ar[ancilla_index])
            for q1,q2 in qubit_pairs[::-1]:
                ev_circuit.cx(q1,q2)
            # undo change-of-basis
            for q,P in non_identity_indices:
                q = self.reverse_qubit_order(q)
                if P != 'Z':
                    ev_circuit.h(qr[q])
                if P == 'Y':
                    ev_circuit.s(qr[q])
        
        # loschmidt echo
        ev_circuit.compose(self.primitive_qc.inverse(), qubits=qr, inplace=True)
        if self.apply_cancellation:
            ev_circuit = self.cancel_inverse_gates(ev_circuit)
        return ev_circuit
    
    def build_echo_verification_circuits(self, paulistring_measurement_dict):
        """
        """
        self.ev_flag = True
        self.paulistring_measurement_dict = paulistring_measurement_dict
        assert paulistring_measurement_dict.keys() == self.clusters.keys(), 'Keys are not equal between the paulistring measurement dict and cluster dict'
        for index, paulistrings_to_measure in paulistring_measurement_dict.items():
            self.echo_verification_circuits[index] = self.prepare_echo_verification(paulistrings_to_measure, index)

    def get_active_circuit(self, index):
        """
        """
        if self.ev_flag:
            return self.echo_verification_circuits[index]
        else:
            return self.primitive_qc

    def _transpile_cluster(self, index):
        """
        """
        assert index < self.n_clusters, 'Index out of range'
        
        circuit = self.get_active_circuit(index)
        if self.ev_flag:
            coupling_graph = self.clusters[index]['graphs']['full']
            n_qubits = self.clusters[index]['n_qubits']['full']
        else:
            coupling_graph = self.clusters[index]['graphs']['compute']
            n_qubits = self.clusters[index]['n_qubits']['compute']

        assert circuit.num_qubits <= n_qubits, 'Circuit contains more qubits than in the cluster.'

        transpiled_circuit = transpile(
            circuit, 
            optimization_level=self.optimization_level, 
            basis_gates=self.config.basis_gates, 
            coupling_map=CouplingMap(coupling_graph.edge_list())
        )
        if not set(
                graph_from_circuit(transpiled_circuit, n_qubits=n_qubits).edge_list()
            ).issubset(coupling_graph.edge_list()):
            raise RuntimeError(f'The circuit layout has not been mapped correctly for cluster {index}.')
    
        return transpiled_circuit
    
    def transpile_clusters(self):
        """
        """
        for i in self.clusters.keys():
            self.transpiled_circuits[i] = self._transpile_cluster(i)

    def locate_in_circuit(self, index):
        """
        """
        sub_qc = self.transpiled_circuits[index]
        layout = sorted(sub_qc.layout.initial_layout.get_physical_bits().items())
        layout = [self.get_active_circuit(index).find_bit(qubit) for _,qubit in layout]
        return layout

    def _embed_cluster(self, index):
        """
        """
        assert self.sys_qc is not None, 'Have not initialized the system circuit.'
        sub_qc = self.transpiled_circuits[index]
        active_qubits = [self.clusters[index]['qubits']['full'][i] for i in get_active_qubits(sub_qc)]
        
        self.ancilla_positions[index] = []
        self.compute_positions[index] = []
        for after,before in self.transpiled_circuits[index].layout.initial_layout.get_physical_bits().items():
            mapped_qubit = self.clusters[index]['qubits']['full'][after]
            if mapped_qubit in active_qubits:
                if before.register.name == 'sys':
                    self.compute_positions[index].append(mapped_qubit)
                elif before.register.name == 'anc':
                    self.ancilla_positions[index].append(mapped_qubit)
        self.sys_qc.compose(sub_qc, qubits=self.clusters[index]['qubits']['full'], inplace=True)
        return sub_qc        

    def embed_clusters(self):
        """
        """
        self.sys_qc = QuantumCircuit(self.config.num_qubits)
        self.ancilla_positions = {}
        self.compute_positions = {}
        for i in self.clusters.keys():
            self._embed_cluster(i)
        self.sys_qc = transpile(
            self.sys_qc,
            optimization_level=0, 
            basis_gates=self.config.basis_gates, 
            coupling_map=self.config.coupling_map
        )
        reverse_qubit_map = {
            self.sys_qc.find_bit(q_trg).index:q_src for q_src,q_trg 
            in self.sys_qc.layout.initial_layout.get_physical_bits().items()
        }
        for i,com in self.compute_positions.items():
            self.compute_positions[i] = [reverse_qubit_map[j] for j in com]
        for i,anc in self.ancilla_positions.items():
            self.ancilla_positions[i] = [reverse_qubit_map[j] for j in anc]
        self._validate_embedded_circuit()

    def _validate_embedded_circuit(self):
        """ Sequence of checks to verify whether cluster embedding was successful
        """
        # check circuit has been embedded
        assert self.sys_qc is not None, 'Have not embedded the cluster circuits.'
        # check transpilation did not increase circuit depth (indicative of qubit routing problem)
        max_cluster_depth = max([self.transpiled_circuits[i].depth() for i in range(self.n_clusters)])
        embedded_depth = self.sys_qc.depth()
        if embedded_depth > max_cluster_depth:
            raise RuntimeError(f'The embedded depth {embedded_depth} exceeds the largest cluster depth {max_cluster_depth}, something has gone wrong in the transpilation.')
        # check number of active qubits matches those in utilized cluster
        active_qubits = list(get_active_qubits(circuit=self.sys_qc))
        if len(active_qubits) > (
                sum([cluster['n_qubits']['compute'] for cluster in self.clusters.values()]) + 
                sum([len(paulis) for paulis in self.paulistring_measurement_dict.values()])
            ):
            raise RuntimeError('Number of qubits in transpiled circuit exceeds those in the specified clusters')
        # check embedded circuit is subgraph isomorphic to the device topology
        if not set(
                graph_from_circuit(circuit=self.sys_qc, n_qubits=self.config.num_qubits).edge_list()
            ).issubset(self.coupling_graph().edge_list()):
            raise RuntimeError('The embedded coupling graph is not subgraph isomorphic to the full topology')
        if not list(map(len, self.ancilla_positions.values())) == list(map(len, self.paulistring_measurement_dict.values())):
            raise RuntimeError('Number of active ancilla does not match number of measured Pauli strings.')

    
    
