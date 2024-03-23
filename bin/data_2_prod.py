import math
import random

import numpy as np

from bin.data_3_part import get_u_list


def generate_state(current_state):
    u_numbers = np.random.uniform(0, 2 * np.pi, 12)
    u_list = get_u_list(u_numbers)
    u_matrix = np.kron(u_list[0], np.kron(u_list[1], np.kron(u_list[2], u_list[3])))
    new_state = u_matrix @ current_state @ np.transpose(u_matrix.conj())
    return new_state


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 10

    entanglement_2_qubit = list()
    for _ in range(num_of_quantum_state):
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()
        r4 = random.random()

        s = r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2
        scale = math.sqrt(s)
        a = r1 / scale
        b = r2 / scale
        c = r3 / scale
        d = r4 / scale

        current_part = np.outer(np.array([a + b * 1j, 0, 0, c + d * 1j]),
                                np.array([a + b * 1j, 0, 0, c + d * 1j]).conj())
        entanglement_2_qubit.append(generate_state(current_part))
