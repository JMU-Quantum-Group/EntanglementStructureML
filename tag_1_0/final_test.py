import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from xgboost import XGBClassifier

full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.load('full_sep_labels.npy', allow_pickle=True)

part_3_data = np.load('part_3_states.npy', allow_pickle=True)
part_3_labels = np.load('part_3_labels.npy', allow_pickle=True)

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

test_target_1 = testing_target.copy()
test_target_1[test_target_1 == 2] = 1

test_target_2 = testing_target.copy()
test_target_2[test_target_2 == 1] = 0
test_target_2[test_target_2 == 2] = 1

training_features = matrices.reshape(matrices.shape[0], matrices.shape[1] * matrices.shape[2])
testing_features = test_matrices.reshape(test_matrices.shape[0], test_matrices.shape[1] * test_matrices.shape[2])

tpot_full_3part_pipeline = XGBClassifier(learning_rate=0.5, max_depth=10, min_child_weight=17, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
# Fix random state in exported estimator
if hasattr(tpot_full_3part_pipeline, 'random_state'):
    setattr(tpot_full_3part_pipeline, 'random_state', 42)

tpot_full_3part_pipeline.fit(training_features, training_target_1)
results_full_3part = tpot_full_3part_pipeline.predict(testing_features)

report_full_3part_text = classification_report(test_target_1, results_full_3part, target_names=["Full Sep", "3 Part and 2 Prod"])
print("report_full_3part_text:")
print(report_full_3part_text)

tpot_3part_2prod_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    XGBClassifier(learning_rate=0.5, max_depth=6, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(tpot_3part_2prod_pipeline.steps, 'random_state', 14)

tpot_3part_2prod_pipeline.fit(training_features, training_target_2)
results_3part_2prod = tpot_3part_2prod_pipeline.predict(testing_features)

report_3part_2prod_text = classification_report(test_target_2, results_3part_2prod, target_names=["Full Sep and 3 Part", "2 Prod"])
print("report_3part_2prod_text:")
print(report_3part_2prod_text)

final_result = np.where(results_3part_2prod == 1, 2, np.where(results_full_3part == 0, 0, 1))
final_report_test = classification_report(testing_target, final_result, target_names=["Full Sep", "3 Part", "2 Prod"])
print("final_report_test:")
print(final_report_test)
