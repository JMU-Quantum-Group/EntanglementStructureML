import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.load('full_sep_labels.npy', allow_pickle=True)

part_3_data = np.load('part_3_pure_states.npy', allow_pickle=True)
part_3_labels = np.load('part_3_pure_labels.npy', allow_pickle=True)

prod_2_data = np.load('prod_2_states.npy', allow_pickle=True)
prod_2_labels = np.load('prod_2_labels.npy', allow_pickle=True)

matrices = np.concatenate((full_sep_data, part_3_data, prod_2_data))
labels = np.concatenate((full_sep_labels, part_3_labels, prod_2_labels))

train_indices = np.arange(matrices.shape[0])
np.random.shuffle(train_indices)
matrices = matrices[train_indices]
training_target = labels[train_indices]

# start test data
full_sep_test_data = np.load('full_sep_states_test.npy', allow_pickle=True)
full_sep_test_labels = np.load('full_sep_labels_test.npy', allow_pickle=True)
part_3_test_data = np.load('part_3_states_test.npy', allow_pickle=True)
part_3_test_labels = np.load('part_3_labels_test.npy', allow_pickle=True)
prod_2_test_data = np.load('prod_2_states_test.npy', allow_pickle=True)
prod_2_test_labels = np.load('prod_2_labels_test.npy', allow_pickle=True)

test_matrices = np.concatenate((full_sep_test_data, part_3_test_data, prod_2_test_data))
test_labels = np.concatenate((full_sep_test_labels, part_3_test_labels, prod_2_test_labels))

indices = np.arange(test_matrices.shape[0])
np.random.shuffle(indices)
test_matrices = test_matrices[indices]
testing_target = test_labels[indices]

training_features = matrices.reshape(matrices.shape[0], matrices.shape[1] * matrices.shape[2])
testing_features = test_matrices.reshape(test_matrices.shape[0], test_matrices.shape[1] * test_matrices.shape[2])

# Average CV score on the training set was: 0.9666538461538462
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=6, min_child_weight=8, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)),
    XGBClassifier(learning_rate=1.0, max_depth=2, min_child_weight=10, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
report_text = classification_report(testing_target, results, target_names=["Full Sep", "3 Part", "2 Prod"])
print(report_text)

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
