"""qmodel - a small quantum-like modeling toolkit.

General-purpose primitives for quantum-like models of cognition and decision:
linear-algebra helpers, state vectors and the Born rule, projectors and events,
measurement (probabilities and state collapse), sequential events with total
probability and interference, Hermitian operators with spectral decomposition and
unitaries, and a POVM check. The functions are model-agnostic and are meant to be
imported by application notebooks.

Conventions: states are 1-D complex arrays of unit length; events are projectors
(Hermitian, idempotent); probabilities follow the Born rule q = ||P psi||^2; a
sequence of events is applied right-to-left in time.
"""

import numpy as np
from scipy.linalg import expm as _expm   # general matrix exponential (not in numpy)


# ---------------------------------------------------------------------------
# Linear-algebra primitives
# ---------------------------------------------------------------------------
def transpose(matrix):
    """Plain transpose of a matrix or vector."""
    return np.array(matrix, dtype=complex).T


def dagger(matrix):
    """Conjugate transpose (Hermitian adjoint, A†)."""
    return np.array(matrix, dtype=complex).conj().T


def matmul(a, b):
    """Matrix product a @ b."""
    return np.matmul(np.array(a, dtype=complex), np.array(b, dtype=complex))


def inner_product(v1, v2):
    """Inner product <v1|v2>; conjugates the first argument (physics convention)."""
    return np.vdot(np.array(v1, dtype=complex), np.array(v2, dtype=complex))


def outer_product(v1, v2):
    """Outer product |v1><v2|; conjugates the bra so |v><v| is a valid projector
    for complex vectors as well as real ones."""
    return np.outer(np.array(v1, dtype=complex), np.array(v2, dtype=complex).conj())


def norm(vec):
    """Euclidean norm of a vector."""
    return np.linalg.norm(np.array(vec, dtype=complex))


def eig(matrix):
    """General eigenvalues and eigenvectors."""
    return np.linalg.eig(np.array(matrix, dtype=complex))


def trace(matrix):
    """Trace of a square matrix."""
    return np.trace(np.array(matrix, dtype=complex))


def kron(a, b):
    """Kronecker (tensor) product of two arrays."""
    return np.kron(np.array(a, dtype=complex), np.array(b, dtype=complex))


# ---------------------------------------------------------------------------
# States and the Born rule
# ---------------------------------------------------------------------------
def state(amplitudes):
    """Build a state vector as a 1-D complex array."""
    return np.array(amplitudes, dtype=complex).reshape(-1)


def normalize(psi):
    """Return psi scaled to unit length."""
    psi = np.array(psi, dtype=complex).reshape(-1)
    n = np.linalg.norm(psi)
    if n == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return psi / n


def is_normalized(psi, tol=1e-9):
    """True if ||psi||^2 == 1 (total probability)."""
    return abs(np.vdot(psi, psi) - 1.0) < tol


def born_rule(psi, operator=None):
    """Born rule. With no operator, return the per-component probabilities
    |psi_i|^2. With an operator, return the expectation <psi|operator|psi>;
    for a projector P this equals the event probability ||P psi||^2."""
    psi = np.array(psi, dtype=complex).reshape(-1)
    if operator is None:
        return np.abs(psi) ** 2
    return np.real(np.vdot(psi, np.asarray(operator, dtype=complex) @ psi))


# ---------------------------------------------------------------------------
# Projectors and events
# ---------------------------------------------------------------------------
def basis_vector(k, n):
    """k-th standard basis column (0-indexed) in dimension n."""
    e = np.zeros(n, dtype=complex)
    e[k] = 1.0
    return e


def projector_from_vector(v):
    """Rank-1 projector |v><v|; v is normalized internally."""
    v = normalize(v)
    return outer_product(v, v)


def indicator(indices, n):
    """Diagonal selector matrix with ones on `indices` and zeros elsewhere,
    e.g. indicator([3, 4], 5) -> diag[0 0 0 1 1]."""
    d = np.zeros(n, dtype=complex)
    d[list(indices)] = 1.0
    return np.diag(d)


def projector_from_indices(indices, n):
    """Projector onto the span of the listed standard basis vectors. In the
    standard basis this coincides with the indicator selector matrix."""
    return indicator(indices, n)


def projector_in_basis(U, indices):
    """Projector for an event defined in a rotated basis: P = U M U†, where M
    selects the given outcome coordinates."""
    U = np.asarray(U, dtype=complex)
    M = indicator(indices, U.shape[0])
    return U @ M @ dagger(U)


def complement(P):
    """Projector for the negation of an event: I - P."""
    P = np.asarray(P, dtype=complex)
    return np.eye(P.shape[0], dtype=complex) - P


# ---------------------------------------------------------------------------
# Measurement: probabilities and state collapse
# ---------------------------------------------------------------------------
def event_prob(P, psi):
    """Probability of the event with projector P: q = ||P psi||^2."""
    psi = np.array(psi, dtype=complex).reshape(-1)
    return float(np.linalg.norm(np.asarray(P, dtype=complex) @ psi) ** 2)


def collapse(P, psi):
    """Post-measurement (conditional) state: P psi / ||P psi||."""
    psi = np.array(psi, dtype=complex).reshape(-1)
    projected = np.asarray(P, dtype=complex) @ psi
    n = np.linalg.norm(projected)
    if n == 0:
        raise ValueError("Event has zero probability; cannot condition on it.")
    return projected / n


def conditional_prob(P_second, P_first, psi):
    """Conditional probability q(second | first) = ||P_second P_first psi||^2 / ||P_first psi||^2."""
    if event_prob(P_first, psi) == 0:
        raise ValueError("Conditioning event has zero probability.")
    return event_prob(P_second, collapse(P_first, psi))


# ---------------------------------------------------------------------------
# Sequential events, total probability, interference
# ---------------------------------------------------------------------------
def sequential_prob(projectors, psi):
    """Probability of a sequence of events applied right-to-left in time.
    projectors = [P_first, P_second, ...] gives ||... P_second P_first psi||^2.
    Two events: sequential_prob([P_A, P_B], psi) is q(A then B)."""
    v = np.array(psi, dtype=complex).reshape(-1)
    for P in projectors:
        v = np.asarray(P, dtype=complex) @ v
    return float(np.linalg.norm(v) ** 2)


def total_probability(partition, P_target, psi):
    """Total probability of the target event over a measured partition:
    q_T = sum_i ||P_target P_i psi||^2."""
    return sum(sequential_prob([P_i, P_target], psi) for P_i in partition)


def interference(partition, P_target, psi):
    """Interference term = q(target) - q_T(target); zero in classical probability."""
    return event_prob(P_target, psi) - total_probability(partition, P_target, psi)


def order_effects(P_A, P_B, psi):
    """Measurement order effects:
        O1 = q(A then B)  - q(B then A)
        O2 = q(~A then B) - q(B then ~A)
    Both are zero when the two events are compatible (commuting projectors)."""
    notA = complement(P_A)
    O1 = sequential_prob([P_A, P_B], psi) - sequential_prob([P_B, P_A], psi)
    O2 = sequential_prob([notA, P_B], psi) - sequential_prob([P_B, notA], psi)
    return O1, O2


# ---------------------------------------------------------------------------
# Hermitian operators, spectral decomposition, unitaries
# ---------------------------------------------------------------------------
def is_hermitian(H, tol=1e-9):
    """True if H == H†."""
    H = np.asarray(H, dtype=complex)
    return np.allclose(H, dagger(H), atol=tol)


def is_unitary(U, tol=1e-9):
    """True if U† U == I."""
    U = np.asarray(U, dtype=complex)
    return np.allclose(dagger(U) @ U, np.eye(U.shape[0]), atol=tol)


def is_projector(P, tol=1e-9):
    """True if P is Hermitian and idempotent (P P == P)."""
    P = np.asarray(P, dtype=complex)
    return np.allclose(P @ P, P, atol=tol) and is_hermitian(P, tol)


def spectral_decomposition(H):
    """Spectral decomposition of H. Returns (eigenvalues, eigenvectors, components)
    where components[k] = lambda_k * V_k V_k† and sum(components) == H. Uses the
    Hermitian solver (real eigenvalues, orthonormal eigenvectors) when H is Hermitian."""
    H = np.asarray(H, dtype=complex)
    vals, vecs = np.linalg.eigh(H) if is_hermitian(H) else np.linalg.eig(H)
    comps = [vals[k] * outer_product(vecs[:, k], vecs[:, k]) for k in range(len(vals))]
    return vals, vecs, comps


def unitary_from_hamiltonian(H, t=1.0, sign=-1):
    """Unitary generated by a Hermitian H over time t:
        U = sum_k exp(sign * i * lambda_k * t) V_k V_k†.
    The default sign=-1 gives the Schrodinger form U(t) = exp(-i H t); use sign=+1
    for U = exp(+i H t). Falls back to a general matrix exponential if H is not Hermitian."""
    H = np.asarray(H, dtype=complex)
    if is_hermitian(H):
        vals, vecs, _ = spectral_decomposition(H)
        phases = np.exp(sign * 1j * vals * t)
        return vecs @ np.diag(phases) @ dagger(vecs)
    return _expm(sign * 1j * H * t)


# ---------------------------------------------------------------------------
# POVM
# ---------------------------------------------------------------------------
def is_povm(operators, tol=1e-9):
    """True if a set of measurement operators forms a POVM: the effects
    E_i = M_i† M_i are positive semidefinite and sum to the identity."""
    ops = [np.asarray(op, dtype=complex) for op in operators]
    effects = [dagger(M) @ M for M in ops]
    total = sum(effects)
    complete = np.allclose(total, np.eye(total.shape[0], dtype=complex), atol=tol)
    positive = all(np.all(np.linalg.eigvalsh((E + dagger(E)) / 2) >= -tol) for E in effects)
    return bool(complete and positive)
