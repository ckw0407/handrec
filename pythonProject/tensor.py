import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc
import numpy as np

# 定义数据集根路径
data_dir = 'dataset3/'
best_accuracy=0.0
# 定义图像预处理的转换操作
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建 ImageFolder 数据集实例
dataset = ImageFolder(root=data_dir, transform=transform)

# 划分训练集和测试集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

# 创建训练集和测试集的数据加载器
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(57600, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 创建神经网络实例
model = ConvNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置 TensorBoard 日志路径
log_dir = 'logs'
writer = SummaryWriter(log_dir)

# 训练模型
num_epochs = 35
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == 'cuda':
    print('Using GPU for training.')
else:
    print('Using CPU for training.')
for epoch in range(num_epochs):
    # 训练模型
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total

    # 记录训练损失和准确率到 TensorBoard
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)

    # 测试模型
    model.eval()
    test_correct = 0
    test_total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_accuracy = 100.0 * test_correct / test_total

    # 计算并记录 ROC 曲线和 AUC 到 TensorBoard
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    for i in range(4):  # 四分类问题
        fpr, tpr, _ = roc_curve((all_labels == i), all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        writer.add_scalar(f'Test/Class_{i}_ROC_AUC', roc_auc, epoch)

    # 记录测试准确率到 TensorBoard
    writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

    # 打印训练结果
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    # 如果准确率有提升，则保存模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'model/best_model.pt')

# 关闭 TensorBoard 写入器
writer.close()
