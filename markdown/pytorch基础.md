<!-- TOC -->

- [Basic](#basic)
    - [常用包](#常用包)
    - [检查配置](#检查配置)
    - [基本配置](#基本配置)
    - [常用操作](#常用操作)
- [Advanced](#advanced)
    - [RNN/LSTM/GRU](#rnnlstmgru)
    - [packed padded](#packed-padded)
    - [读取 csv 数据](#读取-csv-数据)
    - [使用 tensorboardX](#使用-tensorboardx)
    - [构建 CustomDataset](#构建-customdataset)
    - [Unicode2Ascii](#unicode2ascii)
    - [整数标记变成 one-hot 编码](#整数标记变成-one-hot-编码)
    - [seq2seq 使用 attentin](#seq2seq-使用-attentin)
    - [注意事项](#注意事项)
- [Code](#code)
    - [求导演进](#求导演进)
    - [预训练](#预训练)
    - [多输出的网络](#多输出的网络)
    - [训练模型的最佳代码结构](#训练模型的最佳代码结构)
- [Also see](#also-see)

<!-- /TOC -->

## Basic

### 常用包

```python
# Prelims
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from torchvision import datasets, transforms, utils

import os
import re
import copy
import time
import shutil
import string
import random
import argparse
import collections
import unicodedata

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage import io, transform
```

### 检查配置

```python
torch.__version__               # PyTorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
```

### 基本配置

```python
# reproducible
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1
torch.cuda.manual_seed_all(1)

# 判断是否有CUDA支持
torch.cuda.is_available()

# GPU Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
input_tensor = input_tensor.to(device)
input_tensor = input_tensor.cuda(device)

# 指定程序运行在特定GPU卡上
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 多 gpu 训练
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

# 清除GPU存储
torch.cuda.empty_cache()

# 保存加载模型参数
torch.save(model.state_dict(), 'some_model.pt')
some_model.load_state_dict(torch.load('some_model.pt'))

# 保存整个模型（很少这么做）
torch.save(model, PATH)
model = torch.load(PATH)

# 保存训练时的 checkpoint (.tar)
torch.save({
    'epoch': epoch,
    'modelA_state_dict': modelA.state_dict(),
    'modelB_state_dict': modelB.state_dict(),
    'optimizerA_state_dict': optimizerA.state_dict(),
    'optimizerB_state_dict': optimizerB.state_dict(),
    'loss': loss,
    'embedding': embedding.state_dict()
}, PATH)

# 加载模型
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)
checkpoint = torch.load(PATH)
checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
embedding = checkpoint['embedding']
modelA.eval()  # train()
modelB.eval()  # train()

# 保存 torch.nn.DataParallel 模型
torch.save(model.module.state_dict(), PATH)

# 保存 arguments 到 config.txt
opt = parser.parse_args()
with open("config.txt", "w") as f:
    f.write(opt.__str__())

# 计算模型整体参数量
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
print(num_parameters)
```

### 常用操作

```python
# 求梯度
x = torch.randn(2, 3, requires_grad=True)
y = x.sum()
print(y.grad_fn)  # <SumBackward0 object at 0x1041e85f8>
y.backward()
print(x.grad)  # 2*3 矩阵，全是1

# 维度变化
x.size()                              # return tuple-like object of dimensions
torch.cat(tensor_seq, dim=0)          # concatenates tensors along dim
x.view(a,b,...)                       # reshapes x into size (a,b,...)
x.view(-1,a)                          # reshapes x into size (b,a) for some b
x.transpose(a,b)                      # swaps dimensions a and b
x.permute(*dims)                      # permutes dimensions
x.unsqueeze(dim)                      # tensor with added axis
x.unsqueeze(dim=2)                    # (a,b,c) tensor -> (a,b,1,c) tensor

# 打乱顺序
tensor = tensor[torch.randperm(tensor.size(0))]  # Shuffle the first dimension

# 复制张量
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No

# 拼接张量
# 注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，而torch.stack会新增一维。
# 例如当参数是3个10×5的张量，torch.cat的结果是30×5的张量，而torch.stack的结果是3×10×5的张量。
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)

# 得到非零/零元素
torch.nonzero(tensor)               # Index of non-zero elements
torch.nonzero(tensor == 0)          # Index of zero elements
torch.nonzero(tensor).size(0)       # Number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # Number of zero elements

# 张量扩展
# Expand tensor of shape 64*512 to shape 64*512*7*7.
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)

# train or test
model.train()
model.eval()
```

## Advanced

### RNN/LSTM/GRU

```python
# RNN
self.rnn = nn.RNN(input_size=x, hidden_size=h, num_layers=1,
                  bidirectional=False, batch_first=False, dropout=0, nonlinearity='tanh'/'relu')
inputs = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(num_layers * num_directions, batch, hidden_size)

output, hn = rnn(inputs, h0)
output of shape: (seq_len, batch, num_directions * hidden_size)
hn of shape: (num_layers * num_directions, batch, hidden_size)

# LSTM
self.lstm = nn.LSTM(input_size=x, hidden_size=h, num_layers=1,
                    bidirectional=False, batch_first=False, dropout=0)
inputs = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(num_layers * num_directions, batch, hidden_size)

output, (hn,cn) = lstm(inputs, (h0,c0))
output of shape: (seq_len, batch, num_directions * hidden_size)
hn of shape: (num_layers * num_directions, batch, hidden_size)
cn of shape: (num_layers * num_directions, batch, hidden_size)

# GRU
self.gru = nn.GRU(input_size=x, hidden_size=h, num_layers=1,
                  bidirectional=False, batch_first=False, dropout=0)
inputs = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(num_layers * num_directions, batch, hidden_size)

output, hn = gru(inputs, h0)
output of shape: (seq_len, batch, num_directions * hidden_size)
hn of shape: (num_layers * num_directions, batch, hidden_size)
```

### packed padded
```python
# 保证 rnn parallel，最好把最长的长度单独拿出来
class Net(nn.Module):
    def forward(self, padded_input, input_lengths):
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input,input_lengths,batch_first=True)
        output, hidden = self.lstm(packed_input)
        output, _  = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        return output, hidden
```

### 读取 csv 数据

```python
# 行，列 --> df[]
# 区 域 ---> df.loc[], df.iloc[]
# 单元格 --> df.at[], df.iat[]

#创建一个Dataframe
data=pd.DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('ABCD'))

# 第三行
row3 = df.loc['c']
row3 = df.iloc[2]

# 前三行
row03 = df[0:3]

# 第三列
col3 = df.loc[:,'C']
col3 = df.iloc[:,2]

# 前三列
col03 = df.iloc[:,0:-1]

# 前n列为x值，最后一列为y值时
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
x = torch.from_numpy(np.array(x))
y = torch.from_numpy(np.array(y))
```

### 使用 tensorboardX

```python
from tensorboardX import SummaryWriter

writer = SummaryWriter()

for epoch in range(50):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    writer.add_scalar('loss', loss, epoch)
    writer.add_scalars('y_vs_y_pred', {'y': y,'y_pred':y_pred},epoch)

    writer.add_pr_curve('pr_curve', labels, predictions, epoch)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()
```

### 构建 CustomDataset

```python
from torch.utils.data import Dataset, DataLoader

# 直接读取 csv
xy = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, -1]).view(-1, 1)

# 使用 Dataset class
class DiabetesDataset(Dataset):
    def __init__(self, csv_path):
        self.xy = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(self.xy[:,0:-1])
        self.y_data = torch.from_numpy(self.xy[:, -1]).view(-1, 1)

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = DiabetesDataset('data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=2)
```

### Unicode2Ascii

```python
import unicodedata

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if not unicodedata.combining(c))
```

### 整数标记变成 one-hot 编码

```python
a = torch.tensor([1, 3, 5])
one_hot = torch.zeros(3, 6).scatter_(dim=1, index=a.view(-1, 1), value=1)

a = torch.tensor([[1, 3, 5], [0, 2, 4]])
one_hot = torch.zeros(2, 3, 6).scatter_(dim=2, index=a.view(2, -1, 1), value=1)
```

```python
# numpy 时改变
a = [1, 2, 3, 4, 5]
b = [4, 5, 3, 2]
c = [1, 6]

data = [b, c, a]
data = sorted(data, key=len, reverse=True)
print(data)  # [[1, 2, 3, 4, 5], [4, 5, 3, 2], [1, 6]]
lengths = [len(i) for i in data]
T = lengths[0]  # 5
B = len(lengths)  # 3
V = 8

def convert_to_onehot(seq, T, V):
    out = np.zeros((T, V))   # 5 * 8
    out[np.array(range(len(seq))), np.array(seq)] = 1
    return out

data = [convert_to_onehot(seq, T, V) for seq in data]
data = torch.from_numpy(np.array(data))  # 3 * 5 * 8

sequence = pack_padded_sequence(data,lengths, batch_first=True)
print(sequence.data.size())  # 11*8
```

### seq2seq 使用 attentin

```python
# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Decoder
class DecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(DecoderRNN, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(output_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

# Decoder with attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

### 注意事项

1. 在「nn.Module」的「forward」方法中避免使用 Numpy 代码

1. 将「DataSet」从主程序的代码中分离，其中尽量只包含只读对象，避免修改可变对象。并且高负载操作放在 `__getitem__` 中，如加载图片。

1. 如果可能的话，请使用「Use.detach()」从计算图中释放张量。为了实现自动微分，PyTorch 会跟踪所有涉及张量的操作。请使用「.detach()」来防止记录不必要的操作。

1. 不要使用 `total_loss += loss`，因为这样会导致 loss（可微变量）的累积。使用 `total_loss += float(loss)` 即可。

## Code

### 求导演进

```python
# 通用部分
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt

torch.manual_seed(1)

batch_size = 100
input_data = 1000
output_data = 10
hidden_layer = 100

epoch_num = 50
learning_rate = 1e-3
```

```python
# 方法1：手动求导
x = torch.randn(batch_size, input_data)
w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)
y = torch.randn(batch_size, output_data)

losses = []
for epoch in range(epoch_num):
    h1 = x.mm(w1).clamp(min=0)  # 100 * 100
    y_pred = h1.mm(w2)          # 100 * 10

    loss = (y_pred - y).pow(2).sum()
    losses.append(loss)

    print('[{}] loss: {:.4f}'.format(epoch+1, loss))

    # 手动求导
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)
    grad_w1 = x.t().mm(grad_y_pred.mm(w2.t()).clamp(min=0))

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(losses)
plt.show()
```

```python
# 方法2：loss 自动求导
x = torch.randn(batch_size, input_data, requires_grad=False)
w1 = torch.randn(input_data, hidden_layer, requires_grad=True)
w2 = torch.randn(hidden_layer, output_data, requires_grad=True)
y = torch.randn(batch_size, output_data, requires_grad=False)

losses = []
for epoch in range(epoch_num):
    h1 = x.mm(w1).clamp(min=0)  # 100 * 100
    y_pred = h1.mm(w2)          # 100 * 10

    loss = (y_pred - y).pow(2).sum()
    losses.append(loss)

    print('[{}] loss: {:.4f}'.format(epoch+1, loss))

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

plt.plot(losses)
plt.show()
```

```python
# 方法3：使用 forward backward
x = torch.randn(batch_size, input_data, requires_grad=False)
w1 = torch.randn(input_data, hidden_layer, requires_grad=True)
w2 = torch.randn(hidden_layer, output_data, requires_grad=True)
y = torch.randn(batch_size, output_data, requires_grad=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, inputs, w1, w2):
        x = torch.mm(inputs, w1).clamp(min=0)
        x = torch.mm(x, w2)
        return x

    def backward(self):
        pass

model = Model()

losses = []
for epoch in range(epoch_num):
    y_pred = model(x, w1, w2)

    loss = (y_pred - y).pow(2).sum()
    losses.append(loss)

    print('[{}] loss: {:.4f}'.format(epoch+1, loss))

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

plt.plot(losses)
plt.show()
```

```python
# 方法4：使用 nn.Sequential
x = torch.randn(batch_size, input_data, requires_grad=False)
y = torch.randn(batch_size, output_data, requires_grad=False)


# 或者用 add_model 方法
# model = nn.Sequential()
# model.add_module('Linear1', nn.Linear(input_data, hidden_layer))
# model.add_module('ReLU', nn.ReLU())
# model.add_module('Linear2', nn.Linear(hidden_layer, output_data))

model = nn.Sequential(OrderedDict([
    ('Linear1', nn.Linear(input_data, hidden_layer)),
    ('ReLU', nn.ReLU()),
    ('Linear2', nn.Linear(hidden_layer, output_data))
]))
print(model)

losses = []

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    y_pred = model(x)
    loss = loss_function(y, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss)
    print('[{}] loss: {:.4f}'.format(epoch+1, loss))

plt.plot(losses)
plt.show()
```

```python
# 方法5：class写法
x = torch.randn(batch_size, input_data)
y = torch.randn(batch_size, output_data)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_data, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_data)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = Model()
print(model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(epoch_num):
    y_pred = model(x)
    loss = loss_function(y, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss)
    print('[{}] loss: {:.4f}'.format(epoch+1, loss))

plt.plot(losses)
plt.show()
```

### 预训练

```python
# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)
```

### 多输出的网络

```python
# 对于有多个输出的网络（例如使用一个预训练好的 VGG 网络构建感知损失），我们使用以下模式:

# 我们使用由「torchvision」包提供的预训练模型
# 我们将一个网络切分成三个模块，每个模块由预训练模型中的层组成
# 我们通过设置「requires_grad = False」来固定网络权重
# 我们返回一个带有三个模块输出的 list

class Vgg19(nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg19, self).__init__()
    vgg_pretrained_features = models.vgg19(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()

    for x in range(7):
        self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(7, 21):
        self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(21, 30):
        self.slice3.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
        for param in self.parameters():
            param.requires_grad = False

  def forward(self, x):
    h_relu1 = self.slice1(x)
    h_relu2 = self.slice2(h_relu1)
    h_relu3 = self.slice3(h_relu2)
    out = [h_relu1, h_relu2, h_relu3]
    return out
```

### 训练模型的最佳代码结构

```python
# 使用 prefetch_generator 中的 BackgroundGenerator 来加载下一个批量数据
# 使用 tqdm 监控训练过程，并展示计算效率，这能帮助我们找到数据加载流程中的瓶颈

start_n_iter = 0
start_epoch = 0

# load checkpoint if needed/ wanted
if opt.resume:
ckpt = load_checkpoint(opt.path_to_checkpoint)
net.load_state_dict(ckpt['net'])
start_epoch = ckpt['epoch']
start_n_iter = ckpt['n_iter']
optim.load_state_dict(ckpt['optim'])
print("last checkpoint restored")

# typically we use tensorboardX to keep track of experiments
writer = SummaryWriter(...)

# now we start the main loop
n_iter = start_n_iter
for epoch in range(start_epoch, opt.epochs):
# set models to train mode
net.train()

# use prefetch_generator and tqdm for iterating through data
pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
            total=len(train_data_loader))
start_time = time.time()

# for loop going through dataset
for i, data in pbar:
    # data preparation
    img, label = data
    if use_cuda:
        img = img.cuda()
        label = label.cuda()
    ...

    # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
    prepare_time = start_time-time.time()

    # forward and backward pass
    optim.zero_grad()
    ...
    loss.backward()
    optim.step()
    ...

    # udpate tensorboardX
    writer.add_scalar(..., n_iter)
    ...

    # compute computation time and *compute_efficiency*
    process_time = start_time-time.time()-prepare_time
    pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
        process_time/(process_time+prepare_time), epoch, opt.epochs))
    start_time = time.time()

# maybe do a test pass every x epochs
if epoch % x == x-1:
    # bring models to evaluation mode
    net.eval()
    ...
    #do some tests
    pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader, ...)),
            total=len(test_data_loader))
    for i, data in pbar:
        ...

    # save checkpoint if needed
    ...
```

## Also see

- [Pytorch Tutorials](https://pytorch.org/tutorials/) _(pytorch.org)_
- [Pytorch Document](https://pytorch.org/docs/stable/index.html) _(pytorch.org)_
- [PyTorch for Numpy users](https://github.com/wkentaro/pytorch-for-numpy-users) _(github.com)_
- [PyTorch Cookbook](https://zhuanlan.zhihu.com/p/59205847) _(zhihu.com)_
- [Awesome Pytorch](https://github.com/INTERMT/Awesome-PyTorch-Chinese) _(github.com)_
