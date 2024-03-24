import numpy as np
from tpot import TPOTClassifier

full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.array([0] * full_sep_data.shape[0])

part_3_data = np.load('part_3_pure_states.npy', allow_pickle=True)
part_3_labels = np.array([1] * part_3_data.shape[0])

prod_2_data = np.load('prod_2_states.npy', allow_pickle=True)
prod_2_labels = np.array([1] * prod_2_data.shape[0])

matrices = np.concatenate((full_sep_data, part_3_data, prod_2_data))
labels = np.concatenate((full_sep_labels, part_3_labels, prod_2_labels))

train_indices = np.arange(matrices.shape[0])
np.random.shuffle(train_indices)
matrices = matrices[train_indices]
labels = labels[train_indices]

# start test data
full_sep_test_data = np.load('full_sep_states_test.npy', allow_pickle=True)
full_sep_test_labels = np.array([0] * full_sep_test_data.shape[0])
part_3_test_data = np.load('part_3_states_test.npy', allow_pickle=True)
part_3_test_labels = np.array([1] * part_3_test_data.shape[0])
prod_2_test_data = np.load('prod_2_states_test.npy', allow_pickle=True)
prod_2_test_labels = np.array([1] * prod_2_test_data.shape[0])

test_matrices = np.concatenate((full_sep_test_data, part_3_test_data, prod_2_test_data))
test_labels = np.concatenate((full_sep_test_labels, part_3_test_labels, prod_2_test_labels))

indices = np.arange(test_matrices.shape[0])
np.random.shuffle(indices)
test_matrices = test_matrices[indices]
test_labels = test_labels[indices]

matrices = matrices.reshape(matrices.shape[0], matrices.shape[1] * matrices.shape[2])
test_matrices = test_matrices.reshape(test_matrices.shape[0], test_matrices.shape[1] * test_matrices.shape[2])

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=14)
tpot.fit(matrices, labels)
print(tpot.score(test_matrices, test_labels))
tpot.export('tpot_data_pipeline.py')
