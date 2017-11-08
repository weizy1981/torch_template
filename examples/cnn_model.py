from torchvision.datasets import MNIST
from torchvision import transforms
from torch_template.models import Sequential
from torch_template.layers import Conv2d, MaxPool2d, Linear

lr = 1e-3
batch_size = 100
epochs = 50


def build_model():
    model = Sequential()
    model.add(Conv2d(1, 6, 3, stride=1, padding=1))
    model.add(MaxPool2d(2, 2))
    model.add(Conv2d(1, 6, 5, stride=1, padding=0))
    model.add(MaxPool2d(2, 2))
    model.add(Linear(400, 120))
    model.add(Linear(120, 84))
    model.add(Linear(84, 10))
    return model

if __name__ == '__main__':

    # 张量转换函数
    trans_img = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    testset = MNIST('./data', train=False, transform=trans_img, download=True)

    train_x = trainset.train_data
    train_y = trainset.train_labels

    test_x = testset.test_data
    test_y = testset.test_labels

    lenet = build_model()
    lenet.compile(lr=lr)

    # 训练模型
    lenet.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size)

    # 评估模型
    lenet.evaluate(test_x, test_y)