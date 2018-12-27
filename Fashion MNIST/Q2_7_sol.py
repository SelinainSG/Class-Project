# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 28*28 # pixel
hidden_size =20+20 #  decide by  ourself
num_classes = 10 # cluster into 10 casses
num_epochs = 5
batch_size = 60 # 60 examples
learning_rate = 0.001
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
torch.manual_seed(200)

train_set = torchvision.datasets.FashionMNIST(root='./data_fashion', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data_fashion', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ("T-shirt/top","Trouser","Pullover"," Dress ","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")

# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
net = Net(input_size, hidden_size, num_classes)
net.cuda()
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
##where is the bug
import time
start = time.clock()

grad_collect=[]

for epoch in range(num_epochs):

    grad_collect_batch=torch.zeros(batch_size, input_size).cuda()
    for i, data in enumerate(train_loader,0):
        images, labels = data
        images= images.view(-1,1 * 28 * 28).cuda() # read data by certain format
        images, labels = Variable(images, requires_grad= True), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # append gradient
        gradient=images.grad
        grad_collect_batch+=gradient

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))

    grad_collect_batch=grad_collect_batch/6000
    grad_collect_batch.cpu().numpy()
    grad_collect.append(grad_collect_batch)

end=time.clock()
print("Time used:", start-end)

# organize data into a data table

for i in range(len(grad_collect)):
    temp=np.array(grad_collect[i])
    mean_temp=np.mean(temp,axis=0)
    grad_collect[i]=mean_temp

std_collect=np.array(grad_collect)
std=np.std(std_collect,axis=0)
std_10=np.argsort(std)[-10:]

np.savetxt('gradient.csv', grad_collect, delimiter=',')
print(std_10)



# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1,1* 28 * 28)).cuda() # wrong format
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')