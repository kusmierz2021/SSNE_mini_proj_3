class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,100,3)
        self.pool2 = nn.MaxPool2d(4)
        self.fc0 = nn.Linear(4900,2450)
        self.fc1 = nn.Linear(2450,1225)
        self.fc2 = nn.Linear(1225,600)
        self.fc3 = nn.Linear(600, 200)
        self.fc4 = nn.Linear(200,50)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)#flattenalldimensionsexceptbatch
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net().to(device)
net