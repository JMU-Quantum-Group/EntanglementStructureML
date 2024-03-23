from collections import Counter

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import TensorDataset

torch.manual_seed(100)

# 加载数据
full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.load('full_sep_labels.npy', allow_pickle=True)

part_3_data = np.load('part_3_pure_states.npy', allow_pickle=True)
part_3_labels = np.load('part_3_pure_labels.npy', allow_pickle=True)

# full_sep_data = full_sep_data[:2500]
# full_sep_labels = full_sep_labels[:2500]

matrices = np.concatenate((full_sep_data, part_3_data))
labels = np.concatenate((full_sep_labels, part_3_labels))

# 转换为 PyTorch 张量
matrices = torch.tensor(matrices, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

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
test_labels = test_labels[indices]

test_matrices = torch.tensor(test_matrices, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(matrices, labels)
test_dataset = TensorDataset(test_matrices, test_labels)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# train_size = int(0.7 * matrices.shape[0])
# train_dataset, test_dataset = Data.random_split(dataset, [train_size, matrices.shape[0] - train_size])
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=100)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x


def train(model, total_epoch, optimizer):
    for epoch in range(total_epoch):  # 这里我们只做100个epoch，你可以根据需要调整
        mean_loss = 0.0
        count = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            count += 1

        print(f'Epoch {epoch + 1}/{total_epoch}, Loss: {mean_loss / count}')

        # with torch.no_grad():
        #     rho_w_state = torch.tensor(rho_w_list, dtype=torch.float32)
        #     output_w = model(rho_w_state)
        #     _, predicted_w = torch.max(output_w.data, 1)
        #     # 0.05 start
        #     np_predicted = predicted_w.numpy()
        #     if len(np.where(np_predicted == 1)) > 0 and np.all(
        #             np_predicted[:np.where(np_predicted == 1)[0][0]] == 0) and np.all(
        #         np_predicted[np.where(np_predicted == 1)[0][0]:] == 1):
        #         print(np.where(np_predicted == 1)[0][0] + 5)
        #     else:
        #         print("False")


# 定义输入和输出
input_dim = 256
output_dim = 2

model_list = list()
optimizer_list = list()
model_count = 9
for _ in range(model_count):
    current_model = Net(input_dim, output_dim)
    current_optimizer = torch.optim.Adam(current_model.parameters(), lr=0.001)
    model_list.append(current_model)
    optimizer_list.append(current_optimizer)

# # 创建模型
# model1 = Net(input_dim, output_dim)
# model2 = Net(input_dim, output_dim)
# model3 = Net(input_dim, output_dim)
# model4 = Net(input_dim, output_dim)
# model5 = Net(input_dim, output_dim)
#
# model_list = [model1, model2, model3, model4, model5]
#
# # 定义优化器，包含L2正则化
# optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
# optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
# optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
# optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.001)
# optimizer5 = torch.optim.Adam(model5.parameters(), lr=0.001)
#
# optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# index_list = [1, 2, 4, 8]
# w_state = np.zeros(16)
# for index in index_list:
#     w_state[index] = 0.5
# w_state = np.outer(w_state, w_state)
#
# rho_w_list = list()
# number_list = list()
# for i in range(70):
#     number = 0.05 + i * 0.01
#     rho_w_list.append(number * w_state + ((1 - number) / 16) * np.eye(16))
#     number_list.append(number)

# 训练神经网络

for index in range(5):
    train(model_list[index], 300, optimizer_list[index])


def test_ensemble(models):
    all_predictions = []
    all_labels = list()
    with torch.no_grad():
        for model in models:
            current_label = list()
            model_predictions = []
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                model_predictions.extend(predicted.numpy())
                current_label.extend(target.numpy())
            all_predictions.append(model_predictions)
            all_labels.append(current_label)

    # 对每一个样本进行投票
    final_predictions = []
    for sample_predictions in zip(*all_predictions):
        vote = Counter(sample_predictions).most_common(1)[0][0]
        final_predictions.append(vote)

    return final_predictions, all_predictions


final_pred, all_preds = test_ensemble(model_list)

all_labels = list()
with torch.no_grad():  # 不需要计算梯度
    for data, target in test_loader:
        all_labels.extend(target.numpy())


# model.eval()  # 切换到评估模式
#
# all_preds = []
# all_labels = []
#
# with torch.no_grad():  # 不需要计算梯度
#     for data, target in test_loader:
#         output = model(data)
#         # 计算并打印损失
#         # loss = criterion(output, target)
#         _, predicted = torch.max(output.data, 1)
#         all_preds.extend(predicted.numpy())
#         all_labels.extend(target.numpy())

# 计算精确率
report_text = classification_report(all_labels, final_pred, target_names=["Full Sep", "3 Part"])
print(f'final: {report_text}')

count = 1
for item in all_preds:
    report_text_temp = classification_report(all_labels, item, target_names=["Full Sep", "3 Part"])
    print("model", count)
    print(report_text_temp)
    count += 1

