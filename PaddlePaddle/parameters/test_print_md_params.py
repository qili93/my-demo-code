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

for name, param in model.named_parameters():
    print(name)

# layers.l1.weight
# layers.l1.bias
# layers.l3.weight
# layers.l3.bias
# layers.l3._mean
# layers.l3._variance
# layers.l4.weight
# layers.l4.bias
# layers.l6.weight
# layers.l6.bias
# layers.l6._mean
# layers.l6._variance
# layers.l7.weight
# layers.l7.bias
