import numpy as np


def generate_quantum_state():
    A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    A_dagger = A.conj().T
    rho = np.dot(A, A_dagger)
    rho = rho / np.trace(rho)

    rho_pt = np.array([[rho[0, 0], rho[0, 2], rho[2, 0], rho[2, 2]],
                       [rho[0, 1], rho[0, 3], rho[2, 1], rho[2, 3]],
                       [rho[1, 0], rho[1, 2], rho[3, 0], rho[3, 2]],
                       [rho[1, 1], rho[1, 3], rho[3, 1], rho[3, 3]]])

    if np.all(np.linalg.eigvals(rho_pt) >= 0):
        print("Sep")
    else:
        print("ENTANGLE")

    return rho


# Generate a random density matrix
rho = generate_quantum_state()
