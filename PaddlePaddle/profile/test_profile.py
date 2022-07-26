import numpy as np
import paddle
import paddle.profiler as profiler
import paddle.profiler.utils as utils
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader

def my_on_trace_ready(prof): # 定义回调函数，性能分析器结束采集数据时会被调用
    callback = profiler.export_chrome_tracing('./profiler_output') # 创建导出性能数据到profiler_demo文件夹的回调函数
    callback(prof)  # 执行该导出函数
    prof.summary(sorted_by=profiler.SortedKeys.GPUTotal) # 打印表单，按GPUTotal排序表单项

p = profiler.Profiler(scheduler = [3,14], on_trace_ready=my_on_trace_ready, timer_only=False) # 初始化Profiler对象

class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([100]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, image, label=None):
        return self.fc(image)

def train():
    dataset = RandomDataset(20 * 4)
    simple_net = SimpleNet()
    opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                parameters=simple_net.parameters())
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    step_info = ''
    p.start()
    for i, (image, label) in enumerate(loader()):
        out = simple_net(image)
        loss = F.cross_entropy(out, label)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        opt.minimize(avg_loss)
        simple_net.clear_gradients()
        p.step()
        if i % 10 == 0:
            step_info = p.step_info()
            print("Iter {}: {}".format(i, step_info))
    p.stop()
    # return step_info

if __name__ == '__main__':
    train()
  



