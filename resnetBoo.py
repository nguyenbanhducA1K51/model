import torch
import torch.nn as nn


class ResidualBlockNoBottleneck(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlockNoBottleneck, self).__init__()
        self.expansion = 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.relu = nn.ReLU()
        
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(output_channels)
            )
        else:
            # self.shortcut = nn.Sequential()
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        z = self.block(x)
        z += self.shortcut(x)
        z = self.relu(z)
        return z


class ResidualBlockBottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channels, in_channels, stride=1):
        super(ResidualBlockBottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*4)
        )
        self.relu = nn.ReLU()
        
        if stride != 1 or input_channels != self.expansion*in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=in_channels*4, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(in_channels*4)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        z = self.block(x)
        z += self.shortcut(x)
        z = self.relu(z)
        return z

class ResNet(nn.Module):
    def __init__(self, architecture, input_width, output_num,  block_type=ResidualBlockNoBottleneck, input_channels=3):
        super(ResNet, self).__init__()
        self.architecture = architecture
        self.input_width = input_width
        self.output_num = output_num
        self.current_channels = 64

        # preliminary layer
        self.preliminary = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block_type, 64, self.architecture[0])
        self.layer2 = self.make_layer(block_type, 128, self.architecture[1], 2)
        self.layer3 = self.make_layer(block_type, 256, self.architecture[2], 2)
        self.layer4 = self.make_layer(block_type, 512, self.architecture[3], 2)

        # final layer
        self.final = nn.Sequential(
            nn.Flatten(),
            #instead of calculate in_features=self.current_channels*int(input_width/(2**5))**2,
            nn.Linear(in_features=self.current_channels*int(input_width/(2**5))**2, out_features=output_num),
            nn.Softmax(dim=1)
        )


    def forward(self,x):
        z = self.preliminary(x)
        z = self.maxpool1(z)
        
        z = self.layer1(z)
        print ("3 {}".format(z.size() ))
        z = self.layer2(z)
        print ("4 {}".format(z.size() ))
        z = self.layer3(z)
        print ("5 {} ".format(z.size() ))
        z = self.layer4(z)
        print ("6  {}".format(z.size() ))
        z = self.final(z)
        return z

    def make_layer(self, block_type, in_channels, repetition_num, stride=1):
        layers = []
        for _ in range(repetition_num):
            layers.append(block_type(self.current_channels, in_channels, stride=stride))
            self.current_channels = in_channels*block_type.expansion
            stride = 1
        return nn.Sequential(*layers)


testing_residual_block_input = torch.randn(1,64,56,56)

img = torch.randn(1,3,512,512)    
# Testing ResNet
test_model = ResNet([3,4,6,3],512,10, ResidualBlockBottleneck)
print(test_model(img).size())
