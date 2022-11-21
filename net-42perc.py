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


net= Net().to(device)
net

######################### net 38% #################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,100,3)
        self.bn2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
############### net 38% ####################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,100,3)
        self.bn2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(3)
        self.fc0 = nn.Linear(8100,2450)
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
############### net 44% ###################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,150,3)
        self.conv3 = nn.Conv2d(150, 150, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc0 = nn.Linear(5400,2200)
        self.fc1 = nn.Linear(2200,1100)
        self.fc2 = nn.Linear(1100,500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200,50)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)#flattenalldimensionsexceptbatch
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

########## net 45% #####################################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,150,3)
        self.conv3 = nn.Conv2d(150, 150, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.2)
        self.fc0 = nn.Linear(5400,2200)
        self.fc1 = nn.Linear(2200,1100)
        self.fc2 = nn.Linear(1100,500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200,50)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop(x)
        x = torch.flatten(x,1)#flattenalldimensionsexceptbatch
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

############### net 48% ################################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
##Warstwakonwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0)
##Warstwamaxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,150,3)

        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(150, 150, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv2d(150, 150, 3)
        self.pool4 = nn.MaxPool2d(2)

        self.fc0 = nn.Linear(600,300)
        # self.fc1 = nn.Linear(2200,1100)
        # self.fc2 = nn.Linear(1100,500)
        self.fc3 = nn.Linear(300, 150)
        self.fc4 = nn.Linear(150,50)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = torch.flatten(x,1)#flattenalldimensionsexceptbatch
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

