"""Central module for quantum computing homework utilities."""

import numpy as np


def transpose(matrix):
    """Return the transpose of a matrix or vector."""
    return np.array(matrix, dtype=complex).T


def matmul(a, b):
    """Matrix multiply two arrays."""
    return np.matmul(np.array(a, dtype=complex), np.array(b, dtype=complex))


def inner_product(v1, v2):
    """Compute the inner product of two vectors."""
    return np.vdot(np.array(v1, dtype=complex), np.array(v2, dtype=complex))


def outer_product(v1, v2):
    """Compute the outer product of two vectors."""
    return np.outer(np.array(v1, dtype=complex), np.array(v2, dtype=complex))


def norm(vec):
    """Compute the Euclidean norm of a vector."""
    return np.linalg.norm(np.array(vec, dtype=complex))


def eig(matrix):
    """Return eigenvalues and eigenvectors."""
    return np.linalg.eig(np.array(matrix, dtype=complex))


def trace(matrix):
    """Return the trace of a square matrix."""
    return np.trace(np.array(matrix, dtype=complex))


def kron(a, b):
    """Compute the Kronecker product of two arrays."""
    return np.kron(np.array(a, dtype=complex), np.array(b, dtype=complex))


def born_rule(psi, operator=None):
    """Return probabilities from a state vector or expectation values for an operator."""
    state = np.array(psi, dtype=complex)
    if operator is None:
        return np.abs(state) ** 2
    return np.real(np.vdot(state, operator @ state))


def unitary_from_hamiltonian(H, t=1.0):
    """Return U = exp(i H t) for a Hamiltonian H."""
    return np.linalg.expm(1j * np.array(H, dtype=complex) * t)


def is_povm(operators):
    """Check whether a collection of operators forms a POVM."""
    ops = [np.array(op, dtype=complex) for op in operators]
    total = sum(op.conj().T @ op for op in ops)
    return np.allclose(total, np.eye(total.shape[0], dtype=complex))
