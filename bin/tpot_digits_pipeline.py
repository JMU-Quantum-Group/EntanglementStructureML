import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.load('full_sep_labels.npy', allow_pickle=True)

part_3_data = np.load('part_3_pure_states.npy', allow_pickle=True)
part_3_labels = np.load('part_3_pure_labels.npy', allow_pickle=True)

matrices = np.concatenate((full_sep_data, part_3_data))
labels = np.concatenate((full_sep_labels, part_3_labels))

train_indices = np.arange(matrices.shape[0])
np.random.shuffle(train_indices)

matrices = matrices[train_indices]
training_target = labels[train_indices]

# start test data
full_sep_test_data = np.load('full_sep_states_test.npy', allow_pickle=True)
full_sep_test_labels = np.load('full_sep_labels_test.npy', allow_pickle=True)
part_3_test_data = np.load('part_3_pure_states_test.npy', allow_pickle=True)
part_3_test_labels = np.load('part_3_pure_labels_test.npy', allow_pickle=True)

test_matrices = np.concatenate((full_sep_test_data, part_3_test_data))
test_labels = np.concatenate((full_sep_test_labels, part_3_test_labels))

indices = np.arange(test_matrices.shape[0])
np.random.shuffle(indices)
test_matrices = test_matrices[indices]
testing_target = test_labels[indices]

training_features = matrices.reshape(matrices.shape[0], matrices.shape[1] * matrices.shape[2])
testing_features = test_matrices.reshape(test_matrices.shape[0], test_matrices.shape[1] * test_matrices.shape[2])

# # NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9738749999999999
exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=8, min_child_weight=20, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
# print(testing_target)
# print(results)

index_list = [1, 2, 4, 8]
w_state = np.zeros(16)
for index in index_list:
    w_state[index] = 0.5
w_state = np.outer(w_state, w_state)

rho_w_list = list()
number_list = list()
for i in range(70):
    number = 0.05 + i * 0.01
    rho_w_list.append(np.triu(number * w_state + ((1 - number) / 16) * np.eye(16)))
    number_list.append(number)

rho_w_list = np.array(rho_w_list)
rho_w_list = rho_w_list.reshape(rho_w_list.shape[0], rho_w_list.shape[1] * rho_w_list.shape[2])
results = exported_pipeline.predict(rho_w_list)
print(results)

# report_text = classification_report(testing_target, results, target_names=["Full Sep", "3 Part"])
# print(report_text)
