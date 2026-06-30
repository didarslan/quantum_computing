 """qdynamics - time-evolution layer for the Chapter 6 dynamic models.

Markov (Kolmogorov) and quantum (Schrodinger) dynamics on a one-dimensional
belief scale, following eq. 6.1-6.3 of QCD2-BB. Complements the static primitives
in qmodel.
"""

import numpy as np
from scipy.linalg import expm


def gaussian_initial_state(N, eta, nu, p=0.0):
    """eq. 6.1: Gaussian initial state over levels j = 0..N-1.
    Returns (psi, phi): quantum amplitude (unit norm) and Markov distribution |psi|^2."""
    j = np.arange(N)
    g = np.exp(-((j - eta) / nu) ** 2)
    psi = np.exp(-1j * p * j) * g
    psi = psi / np.linalg.norm(psi)        # normalize so sum |psi_j|^2 = 1
    phi = np.abs(psi) ** 2                  # Markov initial distribution (sums to 1)
    return psi, phi


def intensity_matrix(N, alpha, beta):
    """eq. 6.2: Markov generator K (tridiagonal, columns sum to zero).
    Super-diagonal alpha, sub-diagonal beta, with conserving boundary entries."""
    K = np.zeros((N, N))
    for i in range(N - 1):
        K[i, i + 1] = alpha      # super-diagonal
        K[i + 1, i] = beta       # sub-diagonal
    K[1, 0] = alpha              # left boundary (eq. 6.2 corner)
    K[N - 2, N - 1] = beta       # right boundary
    for c in range(N):           # diagonal so each column sums to zero
        K[c, c] = 0.0
        K[c, c] = -K[:, c].sum()
    return K


def hamiltonian_matrix(N, mu, sigma):
    """eq. 6.3: Hermitian H with linear potential mu_j = mu * j / N on the diagonal
    and constant off-diagonal sigma."""
    j = np.arange(N)
    H = np.diag(mu * j / N).astype(complex)
    for i in range(N - 1):
        H[i, i + 1] = sigma
        H[i + 1, i] = sigma
    return H


def markov_transition(K, t):
    """Transition matrix T(t) = exp(t K)."""
    return expm(t * np.asarray(K, dtype=float))


def quantum_unitary(H, t):
    """Evolution operator U(t) = exp(-i t H)."""
    return expm(-1j * t * np.asarray(H, dtype=complex))


def guilty_indicator(N, guilty_from=51):
    """Diagonal MG selecting 'guilty' levels (indices guilty_from..N-1)."""
    d = np.zeros(N)
    d[guilty_from:] = 1.0
    return np.diag(d)


def prob_guilty_markov(phi, MG):
    """Markov probability of responding 'guilty': sum of phi over guilty levels."""
    return float(np.real(np.ones(len(phi)) @ MG @ phi))


def prob_guilty_quantum(psi, MG):
    """Quantum probability of responding 'guilty': ||MG psi||^2."""
    v = MG @ psi
    return float(np.real(np.vdot(v, v)))
