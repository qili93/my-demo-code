import paddle
import paddle.nn as nn
import numpy

class FCN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ('l1', nn.Conv2D(3, 16, kernel_size=3, padding=1)),
            ('l2', nn.ReLU()),
            ('l3', nn.BatchNorm2D(16, momentum=0.9)),        
            ('l4', nn.Conv2D(16, 16, kernel_size=3, padding=1)),
            ('l5', nn.ReLU()),
            ('l6', nn.BatchNorm2D(16, momentum=0.9)),
            ('l7', nn.Conv2D(16, 16, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        return self.layers(x)

x_np = numpy.ones((1, 3, 40, 40)).astype(numpy.float32)
x = paddle.to_tensor(x_np)

model = FCN()
out = model(x)
print(model.layers['l1'].weight) # 这个数据会被随机Normal初始化
print(model.layers['l1'].bias)
print(out)

