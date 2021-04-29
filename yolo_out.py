from torch import nn
import torch as t
import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchlr.utils_for_yolo import *
import tqdm
class DBL(nn.Module):

    def __init__(self,in_ch,out_ch,kernal_size,padding=0,strides=1):
        super(DBL,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernal_size,stride=strides,padding=padding)
        self.bn=nn.BatchNorm2d(out_ch)
        self.act=nn.LeakyReLU(inplace=False)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn(x)
        x=self.act(x)
        return x

class rest_unit(nn.Module):

    def __init__(self,in_ch,hidden_ch,out_ch,kernel_size1,kernel_size2,padding1=0,padding2=1):
        super(rest_unit,self).__init__()
        self.DBL1=DBL(in_ch,hidden_ch,kernal_size=kernel_size1,padding=padding1)
        self.DBL2 = DBL(hidden_ch, out_ch, kernal_size=kernel_size2, padding=padding2)
    def forward(self,x):
        res=x
        x=self.DBL2(self.DBL1(x))
        return t.add(res,x)


class DarkNet(nn.Module):

    def __init__(self):
        super(DarkNet,self).__init__()
        self.conv_blc1=DBL(in_ch=3,out_ch=32,kernal_size=3,strides=1,padding=1)
        self.res2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            rest_unit(in_ch=64,hidden_ch=32,out_ch=64,kernel_size1=1,kernel_size2=3)
        )
        self.res2_downside= nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            rest_unit(in_ch=128, hidden_ch=64, out_ch=128,kernel_size1=1,kernel_size2=3),
            rest_unit(in_ch=128, hidden_ch=64, out_ch=128, kernel_size1=1, kernel_size2=3)
        )
        self.res8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=256, hidden_ch=128, out_ch=256, kernel_size1=1, kernel_size2=3)
        )
        self.res8_downside = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512,kernel_size1=1,kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512,kernel_size1=1,kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512,kernel_size1=1,kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512,kernel_size1=1,kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=512, hidden_ch=256, out_ch=512, kernel_size1=1, kernel_size2=3)
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            rest_unit(in_ch=1024, hidden_ch=512, out_ch=1024, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=1024, hidden_ch=512, out_ch=1024, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=1024, hidden_ch=512, out_ch=1024, kernel_size1=1, kernel_size2=3),
            rest_unit(in_ch=1024, hidden_ch=512, out_ch=1024, kernel_size1=1, kernel_size2=3)
        )

    def forward(self,x):
        x=self.conv_blc1(x)
        x=self.res2(x)
        x=self.res2_downside(x)
        x1=self.res8(x)
        x2=self.res8_downside(x1)
        x3=self.res4(x2)
        return (x1,x2,x3)

class YOLObody(nn.Module):

    def __init__(self):
        super(YOLObody,self).__init__()
        self.darknet_backbone=DarkNet()
        self.rest4_brunch=nn.Sequential(
            DBL(in_ch=1024,out_ch=1024,kernal_size=1,strides=1),
            DBL(in_ch=1024, out_ch=1024, kernal_size=1, strides=1),
            DBL(in_ch=1024, out_ch=1024, kernal_size=1, strides=1),
            DBL(in_ch=1024, out_ch=1024, kernal_size=1, strides=1),
            DBL(in_ch=1024, out_ch=1024, kernal_size=1, strides=1),
        )
        self.rest4_tile=nn.Sequential(
            DBL(in_ch=1024, out_ch=1024, kernal_size=1, strides=1),
            nn.Conv2d(in_channels=1024,out_channels=255,kernel_size=1,stride=1)
        )
        self.rest4_up=nn.Sequential(
            DBL(in_ch=1024, out_ch=512, kernal_size=1, strides=1),
            nn.Upsample(scale_factor=2)
        )
        self.rest8_downside_brunch=nn.Sequential(
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1),
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1),
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1),
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1),
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1)
        )
        self.rest8_downside_up = nn.Sequential(
            DBL(in_ch=512, out_ch=256, kernal_size=1, strides=1),
            nn.Upsample(scale_factor=2)
        )
        self.rest8_downside_tile = nn.Sequential(
            DBL(in_ch=512, out_ch=512, kernal_size=1, strides=1),
            nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1)
        )
        self.rest8_brunch = nn.Sequential(
            DBL(in_ch=256, out_ch=256, kernal_size=1, strides=1),
            DBL(in_ch=256, out_ch=256, kernal_size=1, strides=1),
            DBL(in_ch=256, out_ch=256, kernal_size=1, strides=1),
            DBL(in_ch=256, out_ch=256, kernal_size=1, strides=1),
            DBL(in_ch=256, out_ch=256, kernal_size=1, strides=1)
        )

    def forward(self,x):

        res8,res8d,res4=self.darknet_backbone(x)

        res4ds=self.rest4_brunch(res4)
        res4dsup=self.rest4_up(res4ds)
        res4_b_out=self.rest4_tile(res4ds)

        res8dds = self.rest8_downside_brunch(t.add(res4dsup,res8d))
        res8ddsup = self.rest8_downside_up(res8dds)
        res8_b_out=self.rest8_downside_tile(res8dds)

        res8ds_out=self.rest8_brunch(t.add(res8,res8ddsup))

        scale_coords(13,res4_b_out[:, 0:4, :, :],(416,416))
        scale_coords(13,res4_b_out[:, 85:89, :, :],(416,416))
        scale_coords(13,res4_b_out[:, 170:174, :, :],(416,416))

        scale_coords(26,res8_b_out[:, 0:4, :, :],(416,416))
        scale_coords(26,res8_b_out[:, 85:89, :, :],(416,416))
        scale_coords(26,res8_b_out[:, 170:174, :, :],(416,416))

        scale_coords(52,res8ds_out[:, 0:4, :, :],(416,416))
        scale_coords(52,res8ds_out[:, 85:89, :, :] ,(416,416))
        scale_coords(52,res8ds_out[:, 170:174, :, :] ,(416,416))

        return (res4_b_out,res8_b_out,res8ds_out)


def IOU(bbox1, bbox2, epsilon=1e-3):

    b1_x1, b1_y1 = bbox1[0], bbox1[1]
    b1_x4, b1_y4 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]

    b2_x1, b2_y1 = bbox2[0], bbox2[1]
    b2_x4, b2_y4 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

    IOU_x1, IOU_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    IOU_x4, IOU_y4 = min(b1_x4, b2_x4), min(b1_y4, b2_y4)

    over_area = (IOU_x4 - IOU_x1) * (IOU_y4 - IOU_y1)
    union_area = (b1_x4 - b1_x1) * (b1_y4 - b1_y1) + (b2_x4 - b2_x1) * (b2_y4 - b2_y1) - over_area
    return over_area / (union_area + epsilon)



class Imagesets(Dataset):
    def __init__(self, root_path,annoatation,transform=None):
        super(Imagesets, self).__init__()
        self.transform=transform
        self.image_set,self.labels=self.load_data(root_path,annoatation)
        self.Path=root_path
    def id_to_image_name(self,root_path,ID):
        ID=str(ID)
        a=12
        if len(list(ID))<12:
            ID="0"*(12-len(list(ID)))+ID+".jpg"
        else:
            ID=ID+".jpg"
        return ID
    def load_data(self,root_path,annoatation):
        data=[]
        label=[]
        root=self.Path
        index=0
        for ano in tqdm.tqdm(annoatation,desc=f"Images {index}"):

            image_id=ano["image_id"]
            image_name=self.id_to_image_name(root_path,image_id)
            image_name=self.Path+"/"+image_name
            if os.path.exists(image_name):
                index+=1
                data.append(image_name)
                bbox=ano["bbox"]
                bbox.append(ano["category_id"])
                label.append(bbox)
        print("total {} images".format(index))
        return np.array(data),np.array(label)

    def name2imageset(self,nameset, label,transform=None):
        imageset = []
        cnt=0
        for i in nameset:
            image=Image.open(os.path.join(self.Path,i))
            w,h=image.size()[0],image.size()[1]
            image = np.array(image.resize((416, 416)))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
                image = image.repeat(3, axis=-1)
            elif image.shape[0] == 1:
                image = image.repeat(3, axis=-1)
            if image.shape[-1] != 3:
                image = np.mean(image, axis=-1)
                image = image.repeat(3, axis=-1)
            scale1=w/416
            scale2=h/416
            label[cnt][0],label[cnt][2]=(label[cnt][0],label[cnt][2])*scale1
            label[cnt][1],label[cnt][3]=(label[cnt][1],label[cnt][3])*scale2
            imageset.append(image)
        return np.array(imageset)

    def __getitem__(self, image_id):
        image, label = self.image_set[image_id], self.labels[image_id]
        print(image,label)
        image=self.name2imageset(image,label)
        if self.transform!=None:
            image=self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_set)

