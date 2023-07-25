import torch.nn as nn
import torch
import re

def MakeLayers(params_list):
    conv_layers = nn.Sequential()
    fc_layers = nn.Sequential()
    c_idx=p_idx=f_idx=1
    for param in params_list:
        if param[1] == 'Conv':
            conv_layers.add_module(param[0], nn.Conv2d(
                param[2][0], param[2][1], param[2][2], param[2][3],param[2][4]))
            if len(param) >=4:
                if param[3] == 'Batchnorm':
                    conv_layers.add_module(
                        'batchnorm'+str(c_idx), nn.BatchNorm2d(param[2][1]))
                if param[3]=='Relu' or (param[3] == 'Batchnorm' and param[4]=='Relu'):
                    conv_layers.add_module('relu'+str(c_idx),nn.ReLU(inplace=True))
                else:
                    conv_layers.add_module('sigmoid'+str(c_idx),nn.Sigmoid())
            if len(param) >=6 :
                if param[3] == 'Batchnorm':
                    if param[5]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[6][0], param[6][1],param[6][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[6][0], param[6][1],param[6][2])
                else:
                    if param[4]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[5][0], param[5][1],param[5][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[5][0], param[5][1],param[5][2])
                p_idx+=1
            c_idx+=1
            
        else:
            fc_layers.add_module(param[0], nn.Linear(param[2][0], param[2][1]))
            if len(param) >= 4:
                if param[3] == 'Dropout':
                    fc_layers.add_module('dropout', nn.Dropout(param[4]))
                if param[3] == 'Relu' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Relu'):
                    fc_layers.add_module('relu'+str(f_idx), nn.ReLU(inplace=True))
                elif param[3] == 'Sigmoid' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Sigmoid'):
                    fc_layers.add_module('sigmoid'+str(f_idx), nn.Sigmoid())
                elif param[3] == 'Softmax' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Softmax'):
                    fc_layers.add_module('softmax'+str(f_idx), nn.Softmax())
            f_idx+= 1
    return conv_layers, fc_layers


class MyNet(nn.Module):
    def __init__(self, params_list):
        super(MyNet, self).__init__()
        self.features, self.classifier = MakeLayers(params_list)

    def forward(self, x):
        if len(self.features) > 0:
            feature = self.features(x)
            linear_input = torch.flatten(feature, 1)
            output = self.classifier(linear_input)
        else:
            output = self.classifier(x)
        return output


LeNet5 = [('conv1', 'Conv', (1, 6, 5, 1,1), 'Sigmoid', 'Maxpool',(2,2,0)), ('conv2', 'Conv', (6, 15, 5, 1,0),'Sigmoid', 'Maxpool',(2,2,0)), 
             ('fc1', 'FC', (16*4*4, 120), 'Dropout',0.2,'Sigmoid'), ('fc2', 'FC', (120, 84), 'Sigmoid'),('fc3', 'FC', (84, 10))]


AlexNet=[('conv1', 'Conv', (3, 64, 11, 4,2), 'Relu', 'Maxpool',(3,2,0)), ('conv2', 'Conv', (64, 192, 5, 1,2),'Relu', 'Maxpool',(3,2,0)), 
            ('conv3', 'Conv', (192, 384, 3, 1,1), 'Relu'), ('conv4', 'Conv', (384, 256, 3, 1,1),'Relu'),
            ('conv5', 'Conv', (256, 256, 3, 1,1), 'Relu', 'Maxpool',(3,2,0)),
            ('fc1', 'FC', (256*6*6, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000), 'Relu')]

VGG16=[('conv1_1', 'Conv', (3, 64, 3, 1,0), 'Batchnorm','Relu'), ('conv1_2', 'Conv', (64, 64, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv2_1', 'Conv', (64, 128, 3, 1,0), 'Relu'), ('conv2_2', 'Conv', (128, 128, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv3_1', 'Conv', (128, 256, 3, 1,0), 'Relu'), ('conv3_2', 'Conv', (256, 256, 3, 1,1),'Relu'), ('conv3_3', 'Conv', (256, 256, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv4_1', 'Conv', (256, 512, 3, 1,0), 'Relu'), ('conv4_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv4_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv5_1', 'Conv', (512, 512, 3, 1,0), 'Relu'), ('conv5_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv5_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('fc1', 'FC', (512*7*7, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000))]


def Seq2Tup(sequen):
    lst=[]
    for tup in sequen._modules.items():
        lst.append(tup)
    return lst

def ExtractParam(string,num):
    str_num_list=re.findall(r'[0-9]+\.?[0-9]*',string[string.find('('):])
    if(num==1):
        return float(str_num_list[0])
    num_list=[0]*num
    idx1=0
    while(idx1<num):
        if(num==5 and idx1==2):
            if(str_num_list[2]==str_num_list[3]):
                num_list[2]=int(str_num_list[2])
            else:
                num_list[2]=int(str_num_list[2]),int(str_num_list[3])
            if(str_num_list[4]==str_num_list[5]):
                num_list[3]=int(str_num_list[4])
            else:
                num_list[3]=int(str_num_list[4]),int(str_num_list[5])
            if(len(str_num_list)==8):
                if(str_num_list[6]==str_num_list[7]):
                    num_list[4]=int(str_num_list[7])
                else:
                    num_list[4]=int(str_num_list[6]),int(str_num_list[7])
            idx1=5
        else:
            num_list[idx1]=int(str_num_list[idx1])
            idx1+=1
    return tuple(num_list)
        
def FunType(string):
    if(string.find('BatchNorm')!=-1):
        return 'Batchnorm'
    elif(string.find('ReLU')!=-1):
        return 'Relu'
    elif(string.find('Sigmoid')!=-1):
        return 'Sigmoid'
    elif(string.find('MaxPool')!=-1):
        return ExtractParam(string,3)
    elif(string.find('Dropout')!=-1):
        return ExtractParam(string,1)
    elif(string.find('Softmax')!=-1):
        return 'Softmax'
def Net2Tuple(net):
    tmp=nn.Sequential()
    net_list=[]
    net_param_list=[]
    for item in net._modules.items():
        if(isinstance(item[1],type(tmp))):
            net_list+=Seq2Tup(item[1])
        else:
            net_list.append(item)
    
    idx=0
    while(idx<len(net_list)):
        layer=[]
        layer.append(net_list[idx][0])
        tostr=str(net_list[idx][1])
        if(tostr.find('Conv')!=-1):
            layer.append('Conv')
            idx+=1
            layer.append(ExtractParam(tostr,5))
            while(idx<len(net_list) and str(net_list[idx][1]).find('Linear')==-1 and str(net_list[idx][1]).find('Conv')==-1):
                if(str(net_list[idx][1]).find('MaxPool')!=-1):
                    layer.append('Maxpool')
                layer.append(FunType(str(net_list[idx][1])))
                idx+=1
            net_param_list.append(tuple(layer))
        else:
            layer.append('FC')
            idx+=1
            layer.append(ExtractParam(tostr,2))
            while(idx<len(net_list) and str(net_list[idx][1]).find('Linear')==-1):
                if(str(net_list[idx][1]).find('Dropout')!=-1):
                    layer.append('Dropout')
                layer.append(FunType(str(net_list[idx][1])))
                idx+=1
            net_param_list.append(tuple(layer))
    return net_param_list

if __name__ == "__main__":
    a=Net2Tuple(net3)
    print(a)