import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNetwork1(nn.Module):
    '''original model taken from homework, for fashion MNIST.
    Number of classes reduced from 10 to 2, 
    number of channels increased from 1 (b/w) to 3 (colour images).'''
    def __init__(self):
        super(MyNeuralNetwork1, self).__init__() 
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) # 32 f-maps
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(3, stride=2) 
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1) # 62*62
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1) # 60*60
        self.pool2 = nn.MaxPool2d(3, stride=2) # features halved to size 30*30 
        self.dropout2 = nn.Dropout2d(0.25)

        # 64*29*29 = 53'824 , 61'504=64*961=64*31*31
        self.fc1 = nn.Linear(61504, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        #flatten
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_img_shape(self):
        return(128,128)
    
    def get_model_name(self):
        return('MyNeuralNetwork1')


class MyNeuralNetwork2(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork2, self).__init__() 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # img.size 64*64=4096 , 32 f-maps of size 60*60=3'600
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1) #
        self.pool1 = nn.MaxPool2d(3, stride=2) # input feature.size halved to 32*32
        self.dropout1 = nn.Dropout2d(0.25)

        # 64*29*29 = 53'824 , 61'504=64*961=64*31*31
        # 32*31*31 = 30'752
        self.fc1 = nn.Linear(30752, 2)
        self.fc2 = nn.Linear(2, 2)


    def forward(self, x):
        # TODO: YOUR CODE HERE
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        #flatten
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_img_shape(self):
        return(64,64)
    
    def get_model_name(self):
        return('MyNeuralNetwork2')
    

class MyNeuralNetwork3(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork3, self).__init__() 
        
        # reduced to 8 feature maps
        self.conv1 = nn.Conv2d(3, 4, 3, stride=1, padding=1) # img.size 64*64=4096 , 8 f-maps
        self.pool1 = nn.MaxPool2d(3, stride=2) # input feature.size halved to 32*32
        self.dropout1 = nn.Dropout2d(0.25)

        # 64*29*29 = 53'824 , 61'504=64*961=64*31*31
        # 32*31*31 = 30'752 for 32 feature, half of it for 16 features
        self.fc1 = nn.Linear(3844, 2)
        self.fc2 = nn.Linear(2, 2)


    def forward(self, x):
        #print('initial input tensor size : ' , [x.size(i) for i in range(4)])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        #flatten
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_img_shape(self):
        return(64,64)
    
    def get_model_name(self):
        return('MyNeuralNetwork3')


class MyNeuralNetwork4(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork4, self).__init__() 
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1) # input feature.size 128*128
        self.pool1 = nn.MaxPool2d(3, stride=2) # input feature.size halved to 64*64
        self.dropout1 = nn.Dropout2d(0.25)

        # no increase in output channels, stick to 32 , maybe already decrease them?
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1) # 30*30
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1) # 28*28
        self.pool2 = nn.MaxPool2d(3, stride=2) # features halved to size 14*14 
        self.dropout2 = nn.Dropout2d(0.25)

        # half of 61'504=64*961=64*31*31
        self.fc1 = nn.Linear(30752, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        #flatten
        #print('right before flatten : ' , [x.size(i) for i in range(4)])
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_img_shape(self):
        return(128,128)
    
    def get_model_name(self):
        return('MyNeuralNetwork4')
    

class MyNeuralNetwork5(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork5, self).__init__() 
        self.conv1 = nn.Conv2d(3, 24, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(24, 24, 3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(3, stride=2) 
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(24, 24, 3, stride=1, padding=1) # 30*30
        self.conv4 = nn.Conv2d(24, 24, 3, stride=1, padding=1) # 28*28
        self.pool2 = nn.MaxPool2d(3, stride=2) # features halved to size 14*14 
        self.dropout2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(23064, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        #flatten
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
 
        return x

    def get_img_shape(self):
        return(128,128)
    
    def get_model_name(self):
        return('MyNeuralNetwork5')
    

class MyNeuralNetwork6(nn.Module):
    def __init__(self):
        '''3 blocks of convolution pooling, dropout.
        Going down to 6x6 feature sizes.'''

        super(MyNeuralNetwork6, self).__init__() 
        self.conv1 = nn.Conv2d(3, 24, 3, stride=1, padding=1) # img.size 128*128=16'384 , 24 f-maps of size 124*124=15'376
        self.pool1 = nn.MaxPool2d(3, stride=2) # input feature.size halved to 62*62
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(24, 24, 3, stride=1, padding=1) # 30*30
        self.pool2 = nn.MaxPool2d(3, stride=2) # features halved to size 15*15
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(24, 24, 3, stride=1, padding=1) # 13*13
        self.pool3 = nn.MaxPool2d(3, stride=2) # features halved to size 6*6
        self.dropout3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(5400, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        #flatten
        #print('right before flatten : ' , [x.size(i) for i in range(4)])
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_img_shape(self):
        return(128,128)
    
    def get_model_name(self):
        return('MyNeuralNetwork6')



