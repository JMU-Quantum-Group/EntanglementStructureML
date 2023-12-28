import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateNet(nn.Module):
    def __init__(self, qubits, output_size):
        super(StateNet, self).__init__()
        self.fc_list = list()
        self.relu_list = list()
        n = 1
        while 2 ** (qubits - n) > output_size:
            self.fc_list.append(nn.Linear(2 ** (qubits - n + 1), 2 ** (qubits - n)).to(device))
            self.relu_list.append(nn.ReLU().to(device))
            n += 1
        self.fc_out = nn.Linear(2 ** (qubits - n), output_size).to(device)

    def forward(self, x):
        out = x
        for i in range(len(self.fc_list)):
            out = self.fc_list[i](out)
            out = self.relu_list[i](out)
        out = self.fc_out(out)
        return out


# 确定网络的输入、隐藏层和输出的大小
# input_size = 784  # 输入层大小
# hidden_size = 500  # 隐藏层大小
# output_size = 10  # 输出层大小
#
# # 实例化网络
# net = StateNet(5, 4).to(device)
# data = torch.randn(100, 32, 32).to(device)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1)
#
# outputs = net(data)

# 训练网络
# for epoch in range(100):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.view(-1, 28 * 28).to(device)
#         labels = labels.to(device)
#
#         # 前向传播
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
