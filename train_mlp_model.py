import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import itertools


# 数据加载与预处理
# 读取 Excel 文件，不设表头
df = pd.read_excel('S+P.xlsx', header=None)

# 替换所有缺失值为 0
df.fillna(0, inplace=True)

# 删除第一行（表头示例或无效数据）
df = df.iloc[1:, :]

# 保证第17列是标签列（索引16）
label_series = df.iloc[:, 16].astype(str)

# 过滤掉非法标签（只保留 0–9 的整数）
df = df[label_series.str.fullmatch(r'[0-9]')]

# 分离特征与标签
labels = df.iloc[:, 16].astype(int).values
features = df.iloc[:, :16].astype(float).values  # 特征列为0~15共16列


scaler = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler.fit_transform(features)
X_train, X_val, y_train, y_val = train_test_split(features_normalized, labels, test_size=0.2, random_state=42)

# 输入形状保持 (batch_size, input_dim=10)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# MLP模型定义
class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# 参数设置
input_dim = 16
hidden_dim = 100
num_classes = 10
model = MLPModel(input_dim, hidden_dim, num_classes)

# 损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0, last_epoch=-1)

# 训练循环
num_epochs = 500
best_val_acc = 0
train_acc_list = []
val_acc_list = []
train_pre_list = []
val_pre_list = []
train_rec_list = []
val_rec_list = []
train_f1_list = []
val_f1_list = []

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_preds = []
    train_true = []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        train_preds.extend(predicted.cpu().numpy())
        train_true.extend(target.cpu().numpy())

    train_acc = train_correct / train_total
    train_pre = precision_score(train_true, train_preds, average='macro', zero_division=0)
    train_rec = recall_score(train_true, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(train_true, train_preds, average='macro', zero_division=0)

    model.eval()
    val_correct = 0
    val_total = 0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_pre = precision_score(val_true, val_preds, average='macro', zero_division=0)
    val_rec = recall_score(val_true, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)

    # 记录指标
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_pre_list.append(train_pre)
    val_pre_list.append(val_pre)
    train_rec_list.append(train_rec)
    val_rec_list.append(val_rec)
    train_f1_list.append(train_f1)
    val_f1_list.append(val_f1)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {val_acc * 100:.2f}%')

# 绘制指标曲线
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_acc_list, label='Train')
plt.plot(epochs, val_acc_list, label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_pre_list, label='Train')
plt.plot(epochs, val_pre_list, label='Validation')
plt.title('Precision')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, train_rec_list, label='Train')
plt.plot(epochs, val_rec_list, label='Validation')
plt.title('Recall')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_f1_list, label='Train')
plt.plot(epochs, val_f1_list, label='Validation')
plt.title('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig('mlp_metrics.png')
plt.show()

# 混淆矩阵
conf_matrix = confusion_matrix(val_true, val_preds, labels=np.arange(10))
plt.figure(figsize=(5, 5))
plt.imshow(conf_matrix, cmap=plt.cm.Blues, vmax=10)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = conf_matrix.max() / 2.
thresh = min(thresh, 10)
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'mlp_confusion_matrix.png'")

# 分类指标柱状图
conf_matrix = np.array(conf_matrix)
num_classes = conf_matrix.shape[0]
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1_score = np.zeros(num_classes)

for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score[i] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

classes = ['Class {}'.format(i) for i in range(num_classes)]
x = np.arange(len(classes))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1 Score')

ax.set_ylabel('Scores')
ax.set_title('Scores by class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('mlp_class_scores.png')
plt.show()

# 保存模型
torch.save(model, 'model_mlp.pth')