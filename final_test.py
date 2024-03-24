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

training_target_1 = training_target.copy()
training_target_1[training_target_1 == 2] = 1

training_target_2 = training_target.copy()
training_target_2[training_target_2 == 1] = 0
training_target_2[training_target_2 == 2] = 1

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

tpot_full_3part_pipeline = XGBClassifier(learning_rate=0.5, max_depth=10, min_child_weight=17, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
# Fix random state in exported estimator
if hasattr(tpot_full_3part_pipeline, 'random_state'):
    setattr(tpot_full_3part_pipeline, 'random_state', 42)

tpot_full_3part_pipeline.fit(training_features, training_target_1)
results_full_3part = tpot_full_3part_pipeline.predict(testing_features)

tpot_3part_2prod_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    XGBClassifier(learning_rate=0.5, max_depth=6, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(tpot_3part_2prod_pipeline.steps, 'random_state', 14)

tpot_3part_2prod_pipeline.fit(training_features, training_target_2)
results_3part_2prod = tpot_3part_2prod_pipeline.predict(testing_features)
