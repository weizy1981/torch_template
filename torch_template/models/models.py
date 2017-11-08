import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

# 构建模型
class Sequential(nn.Module):
    # 定义损失函数
    #criterion = None
    # 定义优化器（梯度下降）
    #optimizer = None
    cpu_mode = True
    batch_size = 1
    num_workers = 1
    shuffle = True
    #sequential = None
    layer_num = 0

    def __init__(self, cpu_mode=True):
        super(Sequential, self).__init__()
        self.cpu_mode = cpu_mode
        if cpu_mode:
            self.cpu()
        else:
            self.cuda()

        self.sequential = nn.Sequential()

    def add(self, layer=None, activation=None):
        if True:
            self.layer_num = self.layer_num + 1
            if layer != None:
                name = 'layer' + str(self.layer_num)
                self.sequential.add_module(name=name, module=layer)
            if activation == 'relu':
                name = 'activation' + str(self.layer_num)
                self.sequential.add_module(name=name, module=nn.ReLU())
            elif activation == 'softmax':
                name = 'activation' + str(self.layer_num)
                self.sequential.add_module(name=name, module=nn.Softmax())
            elif activation == 'sigmoid':
                name = 'activation' + str(self.layer_num)
                self.sequential.add_module(name=name, module=nn.Sigmoid())
            elif activation == None:
                pass

    def forward(self, x):
        out = None
        liners = 0
        fattened = False
        if True:
            for model in self.sequential.modules():
                if type(model) is nn.Linear:
                    liners = liners + 1
                    if liners == 1 and fattened:
                        out = out.view(out.size(0), -1)
                else:
                    fattened = True

                out = model(x)

        return out

    def compile(self, lr):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        # 定义优化器（梯度下降）
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def fit(self, x, y, epochs=2, batch_size=1, num_workers=1, shuffle=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        # 训练模型
        self.train()
        for i in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            # 将输入数据集转换成张量
            x_loader = DataLoader(x, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            y_loader = DataLoader(y, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            for (img, label) in zip(x_loader, y_loader):
                # 如果使用CPU
                if self.cpu_mode:
                    img = Variable(img).cpu()
                    label = Variable(label).cpu()
                # 如果使用GPU
                else:
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()

                # 归零操作
                self.optimizer.zero_grad()

                output = self(img)
                loss = self.criterion(output, label)
                # 反向传播
                loss.backward()
                self.optimizer.step()

                running_loss += loss.data[0]
                _, predict = torch.max(output, 1)
                correct_num = (predict == label).sum()
                running_acc += correct_num.data[0]

            running_loss /= len(x)
            running_acc /= len(x)
            print('[%d/%d] Loss: %.5f, Acc: %.2f' % (i + 1, epochs, running_loss, running_acc * 100))

    def evaluate(self, x, y):
        self.eval()
        testloss = 0.0
        testacc = 0.1
        # 将输入数据集转换成张量
        x_loader = DataLoader(x, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        y_loader = DataLoader(y, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        for (img, label) in zip(x_loader, y_loader):
            # 如果使用CPU
            if self.cpu_mode:
                img = Variable(img).cpu()
                label = Variable(label).cpu()
            # 如果使用GPU
            else:
                img = Variable(img).cuda()
                label = Variable(label).cuda()

            output = self(img)
            loss = self.criterion(output, label)
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            testacc += correct_num.data[0]

        testloss /= len(x)
        testacc /= len(x)
        print('Loss: %.5f, Acc: %.2f' % (testloss, testacc * 100))

    def predict(self, x):
        print('ToBe implement!')