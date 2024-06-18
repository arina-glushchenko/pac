import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#task1

class FirstModel(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, 300)
        self.fc2 = nn.Linear(300, out_ch, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        y = self.relu(h)
        return y

x1 = torch.randn(1, 512)
fm = FirstModel(512, 256)
print("Первая модель: ", fm)
print(fm.forward(x1))
print()

#task2

class SecondModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

x2 = torch.randn(1, 256)
sm = SecondModel()
print("Вторая модель: ", sm)
print(sm.forward(x2))
print()

#task3

class ThirdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

x3 = torch.randn(3, 19, 19)
tm = ThirdModel()
print("Третья модель: ", tm)
print(tm.forward(x3))
print()

#task4

class FourthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fst_m = FirstModel(19*19, 256)
        self.snd_m = SecondModel()

    def forward(self, x):
        x = self.fst_m(x)
        x = self.snd_m(x)
        return x

x4 = torch.randn(1, 19*19)
fom = FourthModel()
print("Четвертая модель: ", fom)
print(fom.forward(x4))
