"""Utilities for the MIS 68D quantum-like modeling homework.

The functions in this module keep the notebooks focused on the homework
questions instead of repeating matrix bookkeeping in every section.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def as_complex(array):
    """Return an array with complex dtype."""
    return np.array(array, dtype=complex)


def transpose(matrix):
    """Return the transpose of a matrix or vector."""
    return as_complex(matrix).T


def dagger(matrix):
    """Return the conjugate transpose."""
    return as_complex(matrix).conj().T


def matmul(a, b):
    """Matrix multiply two arrays."""
    return np.matmul(as_complex(a), as_complex(b))


def inner_product(v1, v2):
    """Compute <v1|v2>."""
    return np.vdot(as_complex(v1), as_complex(v2))


def outer_product(v1, v2=None):
    """Compute |v1><v2|, using v1 for both sides when v2 is omitted."""
    left = as_complex(v1).reshape(-1)
    right = left if v2 is None else as_complex(v2).reshape(-1)
    return np.outer(left, right.conj())


def norm(vec):
    """Compute the Euclidean norm of a vector."""
    return np.linalg.norm(as_complex(vec))


def normalize(vec):
    """Return a unit-length vector."""
    arr = as_complex(vec).reshape(-1)
    value = norm(arr)
    if np.isclose(value, 0):
        raise ValueError("Cannot normalize a zero vector.")
    return arr / value


def eig(matrix):
    """Return eigenvalues and eigenvectors."""
    return np.linalg.eig(as_complex(matrix))


def trace(matrix):
    """Return the trace of a square matrix."""
    return np.trace(as_complex(matrix))


def kron(a, b):
    """Compute the Kronecker product of two arrays."""
    return np.kron(as_complex(a), as_complex(b))


def is_unitary(matrix, atol=1e-9):
    """Check U*U = I."""
    U = as_complex(matrix)
    return np.allclose(dagger(U) @ U, np.eye(U.shape[1]), atol=atol)


def projector(indices, dim):
    """Return a diagonal projector selecting zero-based basis indices."""
    P = np.zeros((dim, dim), dtype=complex)
    P[indices, indices] = 1.0
    return P


def ray_projector(vec):
    """Return the rank-one projector |v><v|."""
    ket = normalize(vec)
    return outer_product(ket)


def born_rule(psi, operator=None):
    """Return probabilities from a state vector or an expectation value."""
    state = as_complex(psi).reshape(-1)
    if operator is None:
        return np.abs(state) ** 2
    P = as_complex(operator)
    return float(np.real(np.vdot(state, P @ state)))


def event_probability(state, P):
    """Compute ||P|psi>||^2."""
    projected = as_complex(P) @ as_complex(state).reshape(-1)
    return float(np.real(np.vdot(projected, projected)))


def sequential_probability(state, *projectors):
    """Compute ||Pn...P2P1|psi>||^2 for an ordered measurement sequence."""
    current = as_complex(state).reshape(-1)
    for P in projectors:
        current = as_complex(P) @ current
    return float(np.real(np.vdot(current, current)))


def conditional_state(state, P):
    """Collapse and normalize a vector state after observing event P."""
    projected = as_complex(P) @ as_complex(state).reshape(-1)
    p = float(np.real(np.vdot(projected, projected)))
    if np.isclose(p, 0):
        raise ValueError("Event has zero probability; conditional state undefined.")
    return projected / np.sqrt(p)


def total_probability(state, partition_projectors, target_projector):
    """Compute sum_i q(A_i then D) for a partition {A_i}."""
    return sum(sequential_probability(state, P, target_projector) for P in partition_projectors)


def interference(state, partition_projectors, target_projector):
    """Return q(D), qT(D), and q(D)-qT(D)."""
    direct = event_probability(state, target_projector)
    total = total_probability(state, partition_projectors, target_projector)
    return direct, total, direct - total


def order_effects(state, A, D):
    """Return q(A,D), q(D,A), and their difference."""
    q_ad = sequential_probability(state, A, D)
    q_da = sequential_probability(state, D, A)
    return q_ad, q_da, q_ad - q_da


def unitary_from_hamiltonian(H, t=1.0, sign=-1):
    """Return U(t) = exp(sign*i*H*t), usually exp(-iHt)."""
    return expm(sign * 1j * as_complex(H) * t)


def unitary_from_spectral(H, t=1.0, sign=-1):
    """Build exp(sign*i*H*t) from the spectral decomposition of Hermitian H."""
    values, vectors = np.linalg.eigh(as_complex(H))
    phases = np.diag(np.exp(sign * 1j * values * t))
    return vectors @ phases @ dagger(vectors)


def two_dim_unitary(theta, phi=0.0):
    """Return the 2D basis-change matrix used for A/B/C question examples."""
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def binary_observable(theta, phi=0.0):
    """Return the +1/-1 observable for a two-outcome question."""
    U = two_dim_unitary(theta, phi)
    plus = ray_projector(U[:, 0])
    minus = ray_projector(U[:, 1])
    return plus - minus


def real_rotation(dim=5, theta1=0.0, theta2=0.0, pairs=((0, 4), (1, 3)), diagonal=None):
    """Build the real-valued rotation matrix used in Chapter 4 examples."""
    V = np.eye(dim, dtype=complex)
    for theta, (i, j) in zip((theta1, theta2), pairs):
        c, s = np.cos(theta), np.sin(theta)
        block = np.eye(dim, dtype=complex)
        block[i, i] = c
        block[j, j] = c
        block[i, j] = -s
        block[j, i] = s
        V = block @ V
    if diagonal is not None:
        V = V @ np.diag(np.array(diagonal, dtype=complex))
    return V


def pure_density(state):
    """Return rho=|psi><psi|."""
    psi = normalize(state)
    return outer_product(psi)


def mixed_density(states, probabilities=None):
    """Return a mixed density operator from state vectors and probabilities."""
    vectors = [normalize(s) for s in states]
    if probabilities is None:
        probabilities = np.ones(len(vectors)) / len(vectors)
    probs = np.array(probabilities, dtype=float)
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1):
        raise ValueError("Probabilities must be non-negative and sum to one.")
    rho = np.zeros((vectors[0].size, vectors[0].size), dtype=complex)
    for p, v in zip(probs, vectors):
        rho += p * outer_product(v)
    return rho


def density_probability(rho, P):
    """Compute Tr(P rho)."""
    return float(np.real(trace(as_complex(P) @ as_complex(rho))))


def density_sequence_probability(rho, P, Q):
    """Compute Tr(Q P rho P Q) for P followed by Q."""
    rho = as_complex(rho)
    P = as_complex(P)
    Q = as_complex(Q)
    return float(np.real(trace(Q @ P @ rho @ P @ Q)))


def density_conditional_state(rho, P):
    """Return P rho P / Tr(P rho)."""
    rho = as_complex(rho)
    P = as_complex(P)
    p = density_probability(rho, P)
    if np.isclose(p, 0):
        raise ValueError("Event has zero probability; conditional density undefined.")
    return P @ rho @ P / p


def is_povm(operators, as_effects=False, atol=1e-9):
    """Check whether measurement operators or effects form a POVM."""
    ops = [as_complex(op) for op in operators]
    if as_effects:
        effects = ops
    else:
        effects = [dagger(op) @ op for op in ops]
    total = sum(effects, np.zeros_like(effects[0]))
    positive = all(np.all(np.linalg.eigvalsh((E + dagger(E)) / 2) >= -atol) for E in effects)
    return positive and np.allclose(total, np.eye(total.shape[0], dtype=complex), atol=atol)


def gaussian_state(n=101, eta=50.0, nu=5.0, momentum=0.0, quantum=True):
    """Create the initial Markov probability vector or quantum amplitude state."""
    x = np.arange(n)
    amp = np.exp(-((x - eta) ** 2) / (4 * nu**2)) * np.exp(1j * momentum * x)
    if quantum:
        return normalize(amp)
    probs = np.abs(amp) ** 2
    return probs / probs.sum()


def intensity_matrix(n=101, alpha=100.0, beta=150.0):
    """Birth-death Markov intensity matrix with reflecting boundaries."""
    K = np.zeros((n, n), dtype=float)
    for j in range(n):
        if j > 0:
            K[j - 1, j] = alpha
        if j < n - 1:
            K[j + 1, j] = beta
        K[j, j] = -K[:, j].sum()
    return K


def hamiltonian_matrix(n=101, mu=300.0, sigma=40.0):
    """Finite-difference Hamiltonian with linear potential."""
    H = np.zeros((n, n), dtype=complex)
    x = np.arange(n)
    H[np.arange(n), np.arange(n)] = mu * x / (n - 1)
    coupling = -(sigma**2) / 2
    for j in range(n - 1):
        H[j, j + 1] = coupling
        H[j + 1, j] = coupling
    return H


def markov_transition(K, t):
    """Return exp(tK)."""
    return expm(t * np.array(K, dtype=float))


def quantum_transition(H, t):
    """Return exp(-iHt)."""
    return unitary_from_hamiltonian(H, t=t, sign=-1)


def rating_projectors(n=101, bins=5):
    """Create diagonal projectors for an evenly spaced rating scale."""
    edges = np.linspace(0, n, bins + 1, dtype=int)
    return [projector(np.arange(edges[i], edges[i + 1]), n) for i in range(bins)]


def markov_response_probability(phi, M):
    """Compute J M phi for a Markov probability state and indicator matrix M."""
    return float(np.real(np.ones(len(phi)) @ (as_complex(M) @ as_complex(phi))))


def quantum_response_probability(psi, M):
    """Compute ||M psi||^2."""
    return event_probability(psi, M)


def temporal_bell_probabilities(t=0.25, g=2.0):
    """Return the simple 2D Temporal Bell probabilities from Box 6.5.2."""
    q_c1 = float(np.sin(g * t) ** 2)
    q_c2 = float(np.sin(g * t) ** 2)
    q_c3 = float(np.sin(2 * g * t) ** 2)
    return {"q(D|C1)": q_c1, "q(D|C2)": q_c2, "q(D|C3)": q_c3}
