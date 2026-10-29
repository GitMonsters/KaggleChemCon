"""CircuitGraph - Adaptation of SpatialTopologyEngine for quantum circuits.

Parses quantum circuits into DAGs where gates are nodes and qubit wires form edges.
Enables topological analysis, pattern detection, and rewrite opportunity finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import json


@dataclass
class GateNode:
    """Represents a gate as a node in the circuit DAG - like SpatialGraphNode"""
    node_id: int
    gate_type: str  # e.g., 'H', 'CX', 'Rz', 'CNOT'
    qubits: Tuple[int, ...]  # qubits this gate acts on
    params: List[float] = field(default_factory=list)
    label: str = ""
    
    # Topology info (like RSIMetadata)
    depth_index: int = 0
    in_degree: int = 0
    out_degree: int = 0
    adjacency_list: List[int] = field(default_factory=list)


@dataclass  
class CircuitDAG:
    """DAG representation of quantum circuit"""
    nodes: Dict[int, GateNode]
    edges: List[Tuple[int, int]]  # (from_node, to_node)
    num_qubits: int
    qubit_wire_ends: Dict[int, int]  # qubit -> last node id on it


class CircuitGraph:
    """
    Adapted from SpatialTopologyEngine.
    Extracts topological structure from quantum circuits for rewrite analysis.
    """
    
    def __init__(self, connectivity: int = 2):  # 2 for quantum gates
        self.connectivity = connectivity
    
    def parse_circuit(self, circuit: Any) -> CircuitDAG:
        """
        Parse a quantum circuit into a DAG.
        Supports pytket Circuit, cirq Circuit, or dict/list representations.
        """
        # Try to detect circuit type
        circuit_type = type(circuit).__name__
        
        if 'Circuit' in circuit_type and hasattr(circuit, 'get_commands'):
            # pytket
            return self._parse_pytket(circuit)
        elif 'Circuit' in circuit_type and hasattr(circuit, 'all_operations'):
            # cirq
            return self._parse_cirq(circuit)
        elif isinstance(circuit, dict):
            return self._parse_dict(circuit)
        elif isinstance(circuit, (list, str)):
            return self._parse_generic(circuit)
        else:
            # Try to convert
            return self._parse_generic(circuit)
    
    def _parse_pytket(self, circuit) -> CircuitDAG:
        """Parse pytket Circuit"""
        try:
            from pytket import Circuit
            commands = circuit.get_commands()
            num_qubits = circuit.n_qubits
            
            nodes = {}
            # Track last gate on each qubit
            last_on_qubit = [-1] * num_qubits
            edges = []
            node_id = 0
            
            for cmd in commands:
                gate_name = cmd.op.type.name
                qubits = tuple(circuit.qubits.index(q) for q in cmd.args)
                params = list(cmd.op.params) if hasattr(cmd.op, 'params') else []
                
                node = GateNode(
                    node_id=node_id,
                    gate_type=gate_name,
                    qubits=qubits,
                    params=params,
                    label=f"{gate_name}{qubits}"
                )
                
                # Add edges from previous gates on same qubits
                for q in qubits:
                    if last_on_qubit[q] >= 0:
                        edges.append((last_on_qubit[q], node_id))
                    last_on_qubit[q] = node_id
                
                nodes[node_id] = node
                node_id += 1
            
            # Compute degrees
            for node_id, node in nodes.items():
                in_deg = sum(1 for e in edges if e[1] == node_id)
                out_deg = sum(1 for e in edges if e[0] == node_id)
                adj = [e[1] for e in edges if e[0] == node_id]
                node.in_degree = in_deg
                node.out_degree = out_deg
                node.adjacency_list = adj
            
            return CircuitDAG(nodes=nodes, edges=edges, num_qubits=num_qubits, 
                            qubit_wire_ends={i: last_on_qubit[i] for i in range(num_qubits)})
        except Exception as e:
            raise ValueError(f"Failed to parse pytket circuit: {e}")
    
    def _parse_cirq(self, circuit) -> CircuitDAG:
        """Parse cirq Circuit"""
        try:
            import cirq
            num_qubits = len(circuit.all_qubits())
            qubit_to_idx = {q: i for i, q in enumerate(sorted(circuit.all_qubits(), key=str))}
            
            nodes = {}
            last_on_qubit = [-1] * num_qubits
            edges = []
            node_id = 0
            
            for moment in circuit:
                for op in moment:
                    gate_name = type(op.gate).__name__
                    qubits = tuple(qubit_to_idx[q] for q in op.qubits)
                    params = []
                    if hasattr(op.gate, 'exponent'):
                        params = [op.gate.exponent]
                    
                    node = GateNode(
                        node_id=node_id,
                        gate_type=gate_name,
                        qubits=qubits,
                        params=params,
                        label=f"{gate_name}{qubits}"
                    )
                    
                    for q in qubits:
                        if last_on_qubit[q] >= 0:
                            edges.append((last_on_qubit[q], node_id))
                        last_on_qubit[q] = node_id
                    
                    nodes[node_id] = node
                    node_id += 1
            
            for node_id, node in nodes.items():
                in_deg = sum(1 for e in edges if e[1] == node_id)
                out_deg = sum(1 for e in edges if e[0] == node_id)
                adj = [e[1] for e in edges if e[0] == node_id]
                node.in_degree = in_deg
                node.out_degree = out_deg
                node.adjacency_list = adj
            
            return CircuitDAG(nodes=nodes, edges=edges, num_qubits=num_qubits,
                            qubit_wire_ends={i: last_on_qubit[i] for i in range(num_qubits)})
        except Exception as e:
            raise ValueError(f"Failed to parse cirq circuit: {e}")
    
    def _parse_dict(self, circuit: dict) -> CircuitDAG:
        """Parse circuit from dict representation"""
        gates = circuit.get('gates', circuit.get('ops', []))
        num_qubits = circuit.get('n_qubits', circuit.get('num_qubits', 0))
        
        nodes = {}
        last_on_qubit = [-1] * num_qubits
        edges = []
        
        for node_id, gate in enumerate(gates):
            gate_type = gate.get('type', gate.get('name', 'Unknown'))
            qubits = tuple(gate.get('qubits', gate.get('args', [])))
            params = gate.get('params', [])
            
            node = GateNode(node_id=node_id, gate_type=gate_type, qubits=qubits, 
                          params=params, label=f"{gate_type}{qubits}")
            
            for q in qubits:
                if q < len(last_on_qubit) and last_on_qubit[q] >= 0:
                    edges.append((last_on_qubit[q], node_id))
                if q < len(last_on_qubit):
                    last_on_qubit[q] = node_id
            
            nodes[node_id] = node
        
        for node_id, node in nodes.items():
            in_deg = sum(1 for e in edges if e[1] == node_id)
            out_deg = sum(1 for e in edges if e[0] == node_id)
            adj = [e[1] for e in edges if e[0] == node_id]
            node.in_degree = in_deg
            node.out_degree = out_deg
            node.adjacency_list = adj
        
        return CircuitDAG(nodes=nodes, edges=edges, num_qubits=num_qubits,
                        qubit_wire_ends={i: last_on_qubit[i] for i in range(num_qubits)})
    
    def _parse_generic(self, circuit) -> CircuitDAG:
        """Try to convert generic circuit"""
        # Try pytket convert
        try:
            from pytket import Circuit
            from pytket.circuit import OpType
            # If it's not already a circuit, try to convert
            if hasattr(circuit, 'to_circuit'):
                c = circuit.to_circuit()
                return self._parse_pytket(c)
        except:
            pass
        return CircuitDAG(nodes={}, edges=[], num_qubits=0, qubit_wire_ends={})
    
    def find_adjacent_pairs(self, dag: CircuitDAG) -> List[Tuple[int, int]]:
        """Find gates that are adjacent on the same qubit (can potentially cancel/commute)"""
        pairs = []
        for i in range(len(dag.nodes) - 1):
            for j in range(i + 1, min(i + 10, len(dag.nodes))):  # Look ahead
                # Check if they share qubits and are adjacent
                if set(dag.nodes[i].qubits) & set(dag.nodes[j].qubits):
                    # Check if there's a direct edge or they're adjacent
                    if (i, j) in dag.edges or any(e[0] == i and e[1] == j for e in dag.edges):
                        pairs.append((i, j))
        return pairs
    
    def find_cancelable_pairs(self, dag: CircuitDAG) -> List[Tuple[int, int]]:
        """Find pairs of gates that might cancel (inverses)"""
        cancelable = []
        pairs = self.find_adjacent_pairs(dag)
        for i, j in pairs:
            n1, n2 = dag.nodes[i], dag.nodes[j]
            if n1.gate_type == n2.gate_type and n1.qubits == n2.qubits:
                # Same gate twice - often cancels
                if n1.gate_type in ['X', 'Y', 'Z', 'H', 'CX', 'CNOT']:
                    cancelable.append((i, j))
            elif self._are_inverses(n1, n2):
                cancelable.append((i, j))
        return cancelable
    
    def _are_inverses(self, n1: GateNode, n2: GateNode) -> bool:
        """Check if two gates are inverses"""
        # Simple heuristics
        if n1.gate_type == 'Rx' and n2.gate_type == 'Rx' and n1.qubits == n2.qubits:
            if len(n1.params) == 1 and len(n2.params) == 1:
                return abs((n1.params[0] + n2.params[0]) % (2*3.14159)) < 1e-10
        if n1.gate_type == 'Rz' and n2.gate_type == 'Rz' and n1.qubits == n2.qubits:
            if len(n1.params) == 1 and len(n2.params) == 1:
                return abs((n1.params[0] + n2.params[0]) % (2*3.14159)) < 1e-10
        return False
    
    def get_stats(self, dag: CircuitDAG) -> Dict[str, Any]:
        """Get circuit statistics"""
        gate_counts = {}
        for node in dag.nodes.values():
            gate_counts[node.gate_type] = gate_counts.get(node.gate_type, 0) + 1
        
        two_qubit = sum(c for g, c in gate_counts.items() if len([n for n in dag.nodes.values() if n.gate_type == g and len(n.qubits) >= 2]) > 0 or 'C' in g or 'CX' in g)
        # Better count
        two_qubit_count = sum(1 for n in dag.nodes.values() if len(n.qubits) >= 2)
        
        return {
            'num_gates': len(dag.nodes),
            'num_qubits': dag.num_qubits,
            'gate_counts': gate_counts,
            'two_qubit_gates': two_qubit_count,
            'num_edges': len(dag.edges)
        }
