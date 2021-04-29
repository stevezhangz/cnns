import torch as t
from torch import nn
from torchsummary import summary
from torch.nn import functional
import copy

class Block_Conv1(nn.Module):
    def __init__(self, in_planes, places, stride=1):
        super(Block_Conv1,self).__init__()
        self.conv1_input_channel = in_planes
        self.output_channel = places
        #defining conv1
        self.conv1 = self.Conv(self.conv1_input_channel, self.output_channel, kernel_size=3, stride=stride, padding=1)

    def Conv(self, in_places, places, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(places),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        return out

class Basic_block(nn.Module):
    def __init__(self,len_list, downsampling=False, stride=1):
        super(Basic_block,self).__init__()

        global IND

        in_plane=len_list[IND-1]
        mid=len_list[IND]
        out_plane=len_list[IND+1]

        self.conv1 = self.Basic_conv(in_plane=in_plane, out_plane=mid, stride=stride, padding=1, kernel_size=3)
        self.conv2 = self.Basic_conv(in_plane=mid, out_plane=out_plane, stride=1, padding=1, kernel_size=3)

        if downsampling == True:
            self.downsample = self.Basic_conv(in_plane=in_plane, out_plane=out_plane, stride=stride,kernel_size=1,padding=0)

        if in_plane != out_plane:
            self.downsample = self.Basic_conv(in_plane=in_plane, out_plane=out_plane, stride=stride,kernel_size=1,padding=0)
            downsampling=True
        self.downsampling=downsampling
        self.relu=nn.ReLU()
        IND+=2

    def Basic_conv(self, in_plane, out_plane, kernel_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_plane, out_channels=out_plane, stride=stride, padding=padding,
                      kernel_size=kernel_size),
            nn.BatchNorm2d(out_plane)
        )
        return conv_layer

    def forward(self,x):
        x1=self.relu(self.conv1(x))
        x1=self.conv2(x1)
        if self.downsampling:
            res=self.downsample(x)
            x1+=res
        return self.relu(x1)


class ResNet(nn.Module):
    def __init__(self,block,block_type,len_list,num_lass):
        super(ResNet,self).__init__()
        global IND
        IND=0
        self.conv1=Block_Conv1(in_planes=3, places=len_list[IND])
        IND=1
        self.layer1=self.makelayer(block=block[0],block_type=block_type,stride=1,len_list=len_list)
        self.layer2 = self.makelayer(block=block[2], block_type=block_type, stride=2, len_list=len_list)
        self.layer3 = self.makelayer(block=block[2], block_type=block_type, stride=2, len_list=len_list)

        self.avg=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(in_features=len_list[-2],out_features=num_lass)



    def makelayer(self,block,block_type,stride,len_list):
        layers=[]
        layers.append(block_type(stride=stride,downsampling=True,len_list=len_list))
        for i in range(1,block):
            layers.append(block_type(len_list=len_list))
        return nn.Sequential(*layers)
    @property
    def optimal_layer_info(self):
        cnt=0
        layer_dict={}
        for i in self.state_dict():
            if "conv" in i and "weight" in i:
                cnt += 1
                if cnt % 2 != 0:
                    layer_dict[i]=self.state_dict()[i].size()
            if "fc.weight" in i:
                layer_dict[i] = self.state_dict()[i].size()
            if "downsample" in i and "weight" in i and len(self.state_dict()[i].size())==4:
                layer_dict[i] = self.state_dict()[i].size()
        return layer_dict


    def forward(self,x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.avg(out3).flatten(1)
        x = self.fc(x)
        return x



def ResNet20(CLASS, len_list=None):
    return ResNet(block=[3, 3, 3], len_list=len_list,num_lass=CLASS, block_type=Basic_block)


def ResNet56(CLASS, len_list=None):
    return ResNet([9, 9, 9], len_list=len_list,num_lass=CLASS, block_type=Basic_block)

def ResNet110(CLASS, len_list=None):
    return ResNet([18, 18, 18], len_list=len_list,num_lass=CLASS, block_type=Basic_block)

def ResNet164(CLASS, len_list=None):
    return ResNet([18, 18, 18], len_list=len_list,num_lass=CLASS, block_type=Basic_block)











