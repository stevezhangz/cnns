import os
import numpy as np
from PIL import Image
import torch as t
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch import optim
import tensorflow as tf
import argparse
import tqdm
parser=argparse.ArgumentParser()
parser.add_argument("-p","--path",default="/home/stevezhangz")
parser.add_argument("-d","--data_name",default="imagenet_example")
parser.add_argument("-l","--label_name",default="cid_to_labels.txt")
parser.add_argument("--Batch_size",default=40)
parser.add_argument("--lr",default=1e-3)
parser.add_argument("--epoches",default=70)
parser.add_argument("--rest_layers",default="50")
args_res=parser.parse_args()
#将数据的格式进行转换
pattern = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
# 定义读取数据的方法
class Imagesets(Dataset):
    def __init__(self, path, data_name, label_name, transform,one_hot=False):
        super(Imagesets, self).__init__()

        self.transform = transform
        data_path = os.path.join(path, data_name)
        label_path = os.path.join(path, label_name)
        datas = os.listdir(data_path)
        self.image_set = []
        for i in range(len(datas)):
            image=np.array(Image.open(os.path.join(data_path, datas[i])).resize((224,224)))
            if len(image.shape)==2:
                image=np.expand_dims(image,axis=-1).repeat(3,axis=-1)
            self.image_set.append(image)
        self.image_set = np.array(self.image_set)
        with open(label_path, "r") as f:
            labels = eval(f.read())
        self.labels = []
        for i in labels.items():
            self.labels.append(i[0])
        self.labels = t.from_numpy(np.array(self.labels))
        if one_hot:
            self.labels = t.zeros(1000, 1000).scatter_(1, self.labels.reshape(-1, 1), 1)
    def __getitem__(self, image_id):
        image, label = self.image_set[image_id], self.labels[image_id]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.image_set)
# 将数据读取
train_set=Imagesets(path=args_res.path,data_name=args_res.data_name,
                    label_name=args_res.label_name,transform=pattern)
train_loader=t.utils.data.DataLoader(
    train_set,
    batch_size=args_res.Batch_size,
    shuffle=True,
)
class Block(nn.Module):
    def __init__(self, inch, hidden_ch, outch, stride=(1, 1), downsample=None):
        super(Block, self).__init__()
        self.tiny_Block=nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=hidden_ch, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=(3, 3), stride=stride),
            nn.BatchNorm2d(hidden_ch),
            nn.Conv2d(in_channels=hidden_ch, out_channels=outch, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=False)
        )
        self.act = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.tiny_Block(x)
        if self.downsample!=None:
            res = self.downsample(res)
        x += res
        return self.act(x)
class RestNets(nn.Module):

    def __init__(self, using_prediction=True,default_res="50",class_n=1000):
        super(RestNets, self).__init__()
        self.class_n = class_n
        self.using_pre = using_prediction
        self.head = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )

        all_restnets={
            "152": [[64, 64, 256, 1, 3], [256, 128, 512, 2, 8], [512, 256, 1024, 2, 36], [1024, 512, 2048, 2, 3]],
            "101":[[64, 64, 256, 1, 3], [256, 128, 512, 2, 4], [512, 256, 1024, 2, 23], [1024, 512, 2048, 2, 3]],
            "50":[[64, 64, 256, 1, 3], [256, 128, 512, 2, 4], [512, 256, 1024, 2, 6], [1024, 512, 2048, 2, 3]]
        }

        self.layer_dim_set=all_restnets[default_res]
        self.m=[]
        for layer_setting_info in self.layer_dim_set:
            layers=[]
            layers.append(Block(inch=layer_setting_info[0],
                                hidden_ch=layer_setting_info[1],
                                outch=layer_setting_info[2],
                                stride=layer_setting_info[3],
                                downsample=nn.Conv2d(
                                               in_channels=layer_setting_info[0],
                                               out_channels=layer_setting_info[2],
                                               stride=layer_setting_info[3],
                                               kernel_size=1
                                    )
                                )
                          )

            for i in range(layer_setting_info[-1]-1):
                layers.append(Block(inch=layer_setting_info[2],hidden_ch=layer_setting_info[1],outch=layer_setting_info[2],stride=1))
            self.m.append(nn.Sequential(*layers))
            del(layers)
        self.avgpool = nn.AvgPool2d(2, 2, padding=1)
        self.Linear = nn.Linear(in_features=32768, out_features=self.class_n)
    def forward(self, x):
        x=self.head(x)
        for i in self.m:
            x=i(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear(x)
        return x

class TinyBlock(nn.Module):
    def __init__(self, inch, hidden_ch, outch, stride=(1, 1), downsample=None):
        super(TinyBlock, self).__init__()
        self.tiny_Block=nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=inch, out_channels=hidden_ch, kernel_size=(3, 3), stride=stride),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=hidden_ch, out_channels=outch, kernel_size=(3, 3), stride=1)
        )
        self.act = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.tiny_Block(x)
        if self.downsample!=None:
            res = self.downsample(res)
        x += res
        return self.act(x)
class TinyRestNets(nn.Module):

    def __init__(self, using_prediction=True,default_res="34",class_n=10):
        super(TinyRestNets, self).__init__()
        self.class_n=class_n
        self.using_pre = using_prediction
        self.head = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )

        all_restnets={
            "18":[[64,64,64,1,2],[64,128,128,2,2],[128,256,256,2,2],[256,512,512,2,2]],
            "34": [[64, 64, 64, 1, 3], [64, 128, 128, 2, 4], [128, 256, 256, 2, 6], [256, 512, 512, 2, 3]]
        }

        self.layer_dim_set=all_restnets[default_res]
        self.m=[]
        for layer_setting_info in self.layer_dim_set:
            layers=[]
            layers.append(TinyBlock(inch=layer_setting_info[0],
                                hidden_ch=layer_setting_info[1],
                                outch=layer_setting_info[2],
                                stride=layer_setting_info[3],
                                downsample=nn.Conv2d(
                                               in_channels=layer_setting_info[0],
                                               out_channels=layer_setting_info[2],
                                               stride=layer_setting_info[3],
                                               kernel_size=1
                                    )
                                )
                          )

            for i in range(layer_setting_info[-1]-1):
                layers.append(TinyBlock(inch=layer_setting_info[2],hidden_ch=layer_setting_info[1],outch=layer_setting_info[2],stride=1))
            self.m.append(nn.Sequential(*layers))
            del(layers)
        self.avgpool = nn.AvgPool2d(2, 2, padding=1)
        self.Linear = nn.Linear(in_features=8192, out_features=self.class_n)
    def forward(self, x):
        x=self.head(x)
        for i in self.m:
            x=i(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear(x)
        return x



# 定义训练方法
def train(model, optimizer, loss_f, datasets, epoches, save_path=None):
    loss = []
    grad = []
    correct_size = 0
    total = 0
    criterion=0
    index=0
    m=nn.LogSoftmax()
    for i in range(epoches):
        for index, data in enumerate(tqdm.tqdm(datasets,desc=f"Epoch {i}, Batch{index}, loss{criterion}")):
            train_x, train_y= data
            pre = model(train_x.cuda())
            pre=m(pre)
            criterion = loss_f(pre, train_y.cuda().long())
            criterion = criterion.sum()
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            if index % 2 == 0:
                loss.append(criterion)
    t.save(model.state_dict(),"model_50.pt")
if __name__=="__main__":
    t.autograd.set_detect_anomaly(True)
    t.set_default_tensor_type("torch.cuda.FloatTensor")
    model = RestNets(default_res=args_res.rest_layers,class_n=1000).cuda()
    #model=TinyRestNets(default_res="34",class_n=1000).cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=args_res.lr)
    loss = nn.NLLLoss()
    train(model, optimizer, loss_f=loss, datasets=train_loader, epoches=args_res.epoches, save_path=None)


            
