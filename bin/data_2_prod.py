import numpy as np

index_list = [1, 2, 4, 8]
w_state = np.zeros(16)
for index in index_list:
    w_state[index] = 0.5
w_state = np.outer(w_state, w_state)
rho_w_state = 0.15 * w_state + 0.85 * np.eye(16)

print(rho_w_state)