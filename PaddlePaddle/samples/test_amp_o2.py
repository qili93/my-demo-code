import paddle
import paddle.nn as nn
import paddle.static as static
import numpy as np
import time

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([1, 32, 32]).astype('float16')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

class LeNet5(nn.Layer):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
          nn.BatchNorm2D(num_features=6),
          nn.ReLU(),
          nn.MaxPool2D(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
          nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
          nn.BatchNorm2D(num_features=16),
          nn.ReLU(),
          nn.MaxPool2D(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape((BATCH_SIZE, -1))
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# set device
paddle.enable_static()
place = paddle.NPUPlace(0)
# place = paddle.CUDAPlace(0)

# program
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()

# model and loss
model = LeNet5()
cost = nn.CrossEntropyLoss()

# inputs
images = static.data(shape=[None, 1, 32, 32], name='image', dtype='float16')
labels = static.data(shape=[None, 1], name='label', dtype='int64')

# foward
outputs = model(images)
loss = cost(outputs, labels)

# optimizer and amp
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    init_loss_scaling=1024,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=True,
    use_fp16_guard=False) # use_fp16_guard 设置为 False，网络的全部 op 执行 FP16 计算
optimizer.minimize(loss)

# static executor
exe = static.Executor(place)
exe.run(startup_program)

# init global scope
optimizer.amp_init(place=place, scope=paddle.static.global_scope())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
train_loader = paddle.io.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

# train
total_step = len(train_loader)
for epoch_id in range(EPOCH_NUM):
    epoch_start = time.time()
    for batch_id, (train_image, train_label) in enumerate(train_loader()):
        # print(f"train_image={train_image}")
        # print(f"train_label={train_label}")
        train_loss = exe.run(main_program,
            feed={images.name: train_image,
                  labels.name: train_label},
            fetch_list=[loss.name],
            use_program_cache=True)
    epoch_end = time.time()
    print(f"Epoch ID: {epoch_id+1}, Train epoch time: {(epoch_end - epoch_start) * 1000} ms, Loss: {train_loss}")
