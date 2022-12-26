import torch
from torch import nn

class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch)-1
        self.model = nn.Sequential()
        for i in range(self.conv_number):
            self.model.add_module('conv{0}'.format(i),nn.Conv2d(ch[i], ch[i+1], 3, 1, 1))

    def forward(self, x):
        y = self.model(x)
        return y


class dconv(nn.Module):
    def __init__(self, dims):
        super(dconv,self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dims[0], dims[-1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[-1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.conv_branch(x)
        return y

class FC_EF(nn.Module):
    def __init__(self, in_ch = 3):
        super(FC_EF, self).__init__()
        self.conv1 = convx2(*[in_ch*2, 16, 16])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = convx2(*[16, 32, 32])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = convx2(*[32, 64, 64, 64])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = convx2(*[64, 128, 128, 128])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv1 = dconv(*[[16,16]])#nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = convx2(*[48, 64]) #convx2(*[256, 128, 128, 64])
        self.deconv2 = dconv(*[[64,64]])#nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = convx2(*[128, 128])   #(*[128, 64, 64, 32])
        self.deconv3 = dconv(*[[128,128]]) #nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = convx2(*[256, 256])   #(*[64, 32, 16])
        self.deconv4 = dconv(*[[256,256]]) #nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8 = convx2(*[384, 2])   #(*[32, 16, 2])
        self.pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(384 , 2, bias=False)

    def forward(self, x1, x2):
        h1 = self.conv1(torch.cat((x1,x2), 1))
        h = self.pool1(h1)
        h2 = self.conv2(h)
        h = self.pool2(h2)
        h3 = self.conv3(h)
        h = self.pool3(h3)
        h4 = self.conv4(h)
        h = self.pool4(h4)

        d1 = self.deconv1(h1)
        r1 = torch.cat((d1, h2), 1)
        d2 = self.conv5(r1)
        d2 = self.deconv2(d2)
        r2 = torch.cat((d2, h3), 1)
        d3 = self.conv6(r2)
        d3 = self.deconv3(d3)
        r3 = torch.cat((d3, h4), 1)
        d4 = self.conv7(r3)
        d4 = self.deconv4(d4)
        r4 = torch.cat((d4, h), 1)

        h = self.pool(r4)
        h = self.pool4(h).view(-1, r4.shape[1])

        #view = self.pool(r3).view(-1, r3.shape[1])
        y = self.fc(h)
        return y

'''
if __name__ == '__main__':
    input=torch.randn((1, 3, 128, 128))
    model = FC_EF()  #参数量：2.022448 计算量： 0.835913984
    output = model(input,input)
    print(output.shape)

    from thop import profile
    sflops, sparams = profile(model, inputs=(input,input))
    print("参数量：{} 计算量： {}".format(sparams/1e+6, sflops/1e+9))
'''