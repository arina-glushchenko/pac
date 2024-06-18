import torch
import torch.nn.functional as F

class Layer:
    def __init__(self, size1, size2):
        self.w = torch.randn(size1, size2)
        self.b = torch.randn(size2)
class CustomBareboneModel:
    def __init__(self):
        self.l1 = Layer(256, 64)
        self.l2 = Layer(64, 16)
        self.l3 = Layer(16, 4)

    def forward(self, x):
        x = F.relu(torch.matmul(x, self.l1.w) + self.l1.b)
        x = torch.tanh(torch.matmul(x, self.l2.w) + self.l2.b)
        x = F.softmax(torch.matmul(x, self.l3.w) + self.l3.b, dim=1)
        return x

x = torch.randn(1, 256)
model = CustomBareboneModel()
print(model.forward(x))