from typing import List, Tuple

import numpy as np
import cirq

import quantum_decomp

def single_qubit(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    G = cirq.MatrixGate(matrix)(target_qubits[0])
    converter = cirq.google.optimizers.ConvertToSycamoreGates()
    SycamoreGates = converter.convert(G)
    return SycamoreGates, []

def two_qubit(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    G = cirq.MatrixGate(matrix)(target_qubits[0], target_qubits[1])
    converter = cirq.google.optimizers.ConvertToSycamoreGates()
    SycamoreGates = converter.convert(G)
    return SycamoreGates, []

def swap_to(
    source: cirq.GridQubit,
    target:cirq.GridQubit,
    converter:cirq.google.optimizers.ConvertToSycamoreGates = None
) -> List[cirq.GridQubit]:
    loc = (source.row, source.col)
    dx = [0, -1, 0, 1]
    dy = [1, 0, -1, 0]
    ops = []
    while loc[0] != target.row or loc[1] != target.col:
        best = (100, -1, -1)
        for dir in range(4):
            r = loc[0]+dx[dir]
            c = loc[1]+dy[dir]
            status = (abs(r-target.row)+abs(c-target.col), r, c)
            best = min(best, status)
        a = cirq.GridQubit(loc[0], loc[1])
        b = cirq.GridQubit(best[1], best[2])
        ops.append(cirq.SWAP(a,b))
        loc = (best[1], best[2])

    if converter:
        ops = converter.convert(ops)

    return ops

def do_swap(
    a: cirq.GridQubit, b: cirq.GridQubit
) -> Tuple[List[cirq.GridQubit], cirq.GridQubit]:
    dx = [0, -1, 0, 1]
    dy = [1, 0, -1, 0]
    best = (100, -1, -1)
    for dir in range(4):
        r = b.row+dx[dir]
        c = b.col+dy[dir]
        status = (abs(r-a.row)+abs(c-a.col), r, c)
        best = min(best, status)

    source = a
    target = cirq.GridQubit(best[1], best[2])

    return swap_to(source, target), target

def random_matrix(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray, swap=True,
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:

    circuit = quantum_decomp.matrix_to_cirq_circuit(matrix)
    old_qubits = [(str(x), x) for x in circuit.all_qubits()]
    old_qubits = sorted(old_qubits)
    old_qubits = [x[1] for x in old_qubits]
    mapping = dict(zip(old_qubits, target_qubits))
    
    print(mapping)

    decomp_ops = []
    ops = circuit.all_operations()

    for op in ops:
        if type(op) == cirq.ops.controlled_operation.ControlledOperation:
            gate = op

            controls = gate.controls
            target = gate.sub_operation.qubits[0]
            matrix = cirq.unitary(gate.sub_operation.gate)

            decomp_ops.extend([x.transform_qubits(mapping) for x in 
                cirq.optimizers.decompose_multi_controlled_rotation(
                    matrix,
                    list(controls),
                    target
                )
            ])
        else:
            decomp_ops.append(op.transform_qubits(mapping))

    swapped_ops = []
    for op in decomp_ops:
        if len(op.qubits) == 2:
            q0 = op.qubits[0]
            q1 = op.qubits[1]

            if swap:
                oplist, target = do_swap(q0, q1)
                adjacentop = op.transform_qubits({q0: target, q1: q1})
                revlist = swap_to(target, q0)

                swapped_ops.extend(oplist)
                swapped_ops.append(adjacentop)
                swapped_ops.extend(revlist)
            else:
                swapped_ops.append(op)
        else:
            swapped_ops.append(op)


    SycamoreGates = cirq.google.optimized_for_sycamore(
        cirq.Circuit(swapped_ops),
        optimizer_type='sycamore',
    )

    return SycamoreGates, []


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """

    if np.all(matrix == np.eye(len(target_qubits))):
        return [], []
    if (len(target_qubits) == 1):
        return single_qubit(target_qubits, matrix)
    if (len(target_qubits) == 2):
        return two_qubit(target_qubits, matrix)
    if np.count_nonzero(matrix) == len(target_qubits):
        # Either diagonal or increment
        if len(target_qubits) < 5:
            return random_matrix(target_qubits, matrix)
        elif len(target_qubits) < 6:
            return random_matrix(target_qubits, matrix, swap=False)
    if len(target_qubits) < 5:
        return random_matrix(target_qubits, matrix)
    elif len(target_qubits) < 5:
        return random_matrix(target_qubits, matrix, swap=False)

    return NotImplemented, []
