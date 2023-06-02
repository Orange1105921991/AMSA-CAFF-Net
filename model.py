import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from functools import reduce

class CAFF(nn.Module):
    def __init__(self,in_channels,M=2,r=16,L=32):
        super(CAFF,self).__init__()
        self.out_channels=in_channels
        d=max(in_channels//r,L)
        self.M=M
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1=nn.Sequential(nn.Conv2d(in_channels,d,1,bias=False),nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,in_channels*M,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, input1,input2,input3):
        batch_size=input1.size(0)
        input=[input1,input2,input3]
        feature_sum=input1+input2+input3

        s=self.global_pool(feature_sum)
        z=self.fc1(s)
        a_b=self.fc2(z)

        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b)

        a_b=list(a_b.chunk(self.M,dim=1))

        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b))
        V=list(map(lambda x,y:x*y,input,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V

class AMSA(nn.Module):
    def __init__(self, features, out_features=512):
        super(AMSA, self).__init__()
        self.weight = nn.Conv2d(features, features, kernel_size=1)

        self.avgpool1=nn.AdaptiveAvgPool2d(output_size=(16, 16))
        self.depthwise1_3 = nn.Conv2d(features, features, 3, padding=1, groups=features)
        self.pointwise1_3 = nn.Conv2d(features, features, kernel_size=1)
        self.depthwise1_5 = nn.Conv2d(features, features, 5, padding=2, groups=features)
        self.pointwise1_5 = nn.Conv2d(features, features, kernel_size=1)


        self.avgpool2=nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.depthwise2_3 = nn.Conv2d(features, features, 3, padding=1, groups=features)
        self.pointwise2_3 = nn.Conv2d(features, features, kernel_size=1)
        self.depthwise2_5 = nn.Conv2d(features, features, 5, padding=2, groups=features)
        self.pointwise2_5 = nn.Conv2d(features, features, kernel_size=1)


        self.avgpool3=nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.depthwise3_3 = nn.Conv2d(features, features, 3, padding=1, groups=features)
        self.pointwise3_3 = nn.Conv2d(features, features, kernel_size=1)
        self.depthwise3_5 = nn.Conv2d(features, features, 5, padding=2, groups=features)
        self.pointwise3_5 = nn.Conv2d(features, features, kernel_size=1)


        self.avgpool4=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.bottleneck = nn.Conv2d(features * 5, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight(weight_feature))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        pool_scale1=self.avgpool1(feats)
        pool_scale1_3 = self.pointwise1_3(self.depthwise1_3(pool_scale1))
        pool_scale1_5 = self.pointwise1_5(self.depthwise1_5(pool_scale1))
        scale1_3 = F.upsample(input=pool_scale1_3, size=(h, w), mode='bilinear')
        scale1_5 = F.upsample(input=pool_scale1_5, size=(h, w), mode='bilinear')
        weight1_3 = self.__make_weight(feats, scale1_3)
        weight1_5 = self.__make_weight(feats, scale1_5)
        scale1=(scale1_3*weight1_3+scale1_5*weight1_5)/(weight1_3+weight1_5)

        pool_scale2 = self.avgpool2(feats)
        pool_scale2_3 = self.pointwise2_3(self.depthwise2_3(pool_scale2))
        pool_scale2_5 = self.pointwise2_5(self.depthwise2_5(pool_scale2))
        scale2_3 = F.upsample(input=pool_scale2_3, size=(h, w), mode='bilinear')
        scale2_5 = F.upsample(input=pool_scale2_5, size=(h, w), mode='bilinear')
        weight2_3 = self.__make_weight(feats, scale2_3)
        weight2_5 = self.__make_weight(feats, scale2_5)
        scale2 = (scale2_3 * weight2_3 + scale2_5 * weight2_5) / (weight2_3 + weight2_5)

        pool_scale3=self.avgpool3(feats)
        pool_scale3_3 = self.pointwise3_3(self.depthwise3_3(pool_scale3))
        pool_scale3_5 = self.pointwise3_5(self.depthwise3_5(pool_scale3))
        scale3_3 = F.upsample(input=pool_scale3_3, size=(h, w), mode='bilinear')
        scale3_5 = F.upsample(input=pool_scale3_5, size=(h, w), mode='bilinear')
        weight3_3 = self.__make_weight(feats, scale3_3)
        weight3_5 = self.__make_weight(feats, scale3_5)
        scale3 = (scale3_3 * weight3_3 + scale3_5 * weight3_5) / (weight3_3 + weight3_5)

        scale4 = self.avgpool4(feats)
        scale4 = F.upsample(scale4, size=(h, w), mode='bilinear')

        overall_features=torch.cat([feats,scale1,scale2,scale3,scale4],dim=1)
        bottle = self.bottleneck(overall_features)
        return self.relu(bottle)

class Vgg10(nn.Module):
    def __init__(self):
        super(Vgg10, self).__init__()
        self.vgg1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vgg3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vgg5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vgg8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.vgg10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True))

    def forward(self, x):
        out11=self.vgg1(x)
        out12=self.vgg2(out11)

        out2=self.maxpool1(out12)
        out21=self.vgg3(out2)
        out22=self.vgg4(out21)

        out3= self.maxpool2(out22)
        out31 = self.vgg5(out3)
        out32 = self.vgg6(out31)
        out33 = self.vgg7(out32)

        out4 = self.maxpool3(out33)
        out4 = self.vgg8(out4)
        out4 = self.vgg9(out4)
        out4 = self.vgg10(out4)

        return out11,out12,out21,out22,out31,out32,out33,out4


class AMSA_CAFFNet(nn.Module):
    def __init__(self, load_weights=False):
        super(AMSA_CAFFNet, self).__init__()
        self.seen = 0
        self.frontend=Vgg10()
        self.context = AMSA(512, 512)

        self.Select_feature1=CAFF(64,3)
        self.Select_feature2=CAFF(128,3)
        self.Select_feature3=CAFF(256,3)

        self.back_end1=nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.back_end2=nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.up_sample1=nn.Upsample(scale_factor=2, mode='bilinear')
        self.back_end3=nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

        self.back_end4=nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.back_end5=nn.Sequential(nn.Conv2d(256,128,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        self.back_end6=nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.up_sample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.back_end7=nn.Sequential(nn.Conv2d(128,64,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.back_end8=nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.select31=nn.Conv2d(256,256,kernel_size=1,padding=0,dilation=1)
        self.select32=nn.Conv2d(256,256,kernel_size=1,padding=0,dilation=1)
        self.select21=nn.Conv2d(128,128,kernel_size=1,padding=0,dilation=1)
        self.select22=nn.Conv2d(128,128,kernel_size=1,padding=0,dilation=1)
        self.select11=nn.Conv2d(64,64,kernel_size=1,padding=0,dilation=1)
        self.select12=nn.Conv2d(64,64,kernel_size=1,padding=0,dilation=1)

        self.skip3=nn.Conv2d(512,256,kernel_size=1,padding=0,dilation=1)
        self.skip2 = nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1)
        self.skip1 = nn.Conv2d(128, 64, kernel_size=1, padding=0, dilation=1)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,x):
        out11,out12,out21,out22,out31,out32,out33,out4 = self.frontend(x)

        out = self.context(out4)
        f3=self.skip3(out)
        out=self.back_end1(out)
        out=self.back_end2(out)
        out=self.up_sample1(out)
        out=self.back_end3(out)+F.interpolate(f3,scale_factor=2,mode='bilinear')

        f2 = self.skip2(out)
        out=self.Select_feature3(out,self.select31(out31),self.select32(out32))
        out=self.back_end4(out)

        out=self.up_sample2(out)
        out=self.back_end5(out)+F.interpolate(f2,scale_factor=2,mode='bilinear')

        f1 = self.skip1(out)
        out = self.Select_feature2(out, self.select21(out21), self.select22(out22))
        out = self.back_end6(out)

        out = self.up_sample3(out)
        out = self.back_end7(out)+F.interpolate(f1,scale_factor=2,mode='bilinear')

        out = self.Select_feature1(out, self.select11(out11), self.select12(out12))
        out=self.back_end8(out)
        out=self.output_layer(out)

        return F.sigmoid(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


