# Load in relevant libraries, and alias where appropriate
import time
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

EPOCH_NUM = 5
BATCH_SIZE = 4096

device = torch.device('npu:0')

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# set device
torch.npu.set_device('npu:0')
# device = torch.device('npu:0')

# model
# model = LeNet5().to(device)
model = LeNet5().to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# conver to amp model
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024, verbosity=1, combine_grad=False)

# data loader
transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))])

#Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           drop_last=True,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = BATCH_SIZE,
                                           drop_last=True,
                                           shuffle = True)

# train
total_step = len(train_loader)
for epoch_id in range(EPOCH_NUM):
    epoch_start = time.time()
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # enable graph mode
        torch.npu.enable_graph_mode()

        # if epoch_id ==4 and batch_id == 4:
        #     torch.npu.prof_init("./profile_lenet5_epoch4_iter4")
        #     torch.npu.prof_start()

        # if epoch_id ==4 and batch_id >= 4 and batch_id <=500:
        #     torch.npu.iteration_start()
        
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.npu.launch_graph()

        # if epoch_id ==4 and batch_id >= 4 and batch_id <=500:
        #     torch.npu.iteration_end()

        # if epoch_id ==4 and batch_id == 4:
        #     torch.npu.prof_stop()
        #     torch.npu.prof_finalize()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_id+1, EPOCH_NUM, batch_id+1, total_step, loss.item()))

    torch.npu.disable_graph_mode()

    torch.npu.synchronize()
    epoch_end = time.time()
    print(f"Epoch ID: {epoch_id+1}, Train epoch time: {(epoch_end - epoch_start) * 1000} ms")
