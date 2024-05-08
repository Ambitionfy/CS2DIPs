import torch
import torch.nn as nn
import numpy as np
from common import *
from mmcv.ops import DeformConv2dPack as DCN

class Upsamplefun(nn.Module):
    def __init__(self, scale_factor, mode='2D', fun='Upsample', num_channels=None, upsample_mode=None):
        super(Upsamplefun, self).__init__()
        if fun == 'Upsample':
            self.up = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)
        elif fun == 'PixelShuffle':
            if mode=='2D':
                self.up = nn.Sequential(
                    conv(num_channels, num_channels*(scale_factor**2), 1, bias=True, pad='reflection'),
                    nn.PixelShuffle(2))
            elif mode == '1D':
                self.up = nn.Sequential(
                    conv1d(num_channels, num_channels*(scale_factor**2), 1, bias=True, pad='reflection'),
                    nn.PixelShuffle(2))
    def forward(self, x):
        Output = self.up(x)
        return Output


class SAU(nn.Module):
    # Spatial_Attention_Unit
    def __init__(self, num_channels_down, mode='2D', act_fun='LeakyReLU', upsample_mode = 'bilinear'):
        super(SAU, self).__init__()
        if mode == '2D':
            self.SA1 = nn.Sequential(
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun),
                )

            self.SA2 = nn.Sequential(
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun),
                )

            self.zconv21 = nn.Sequential(
                Upsamplefun(scale_factor=2, fun='Upsample', num_channels=num_channels_down, upsample_mode=upsample_mode),
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun)
            )
            self.zconv22 = nn.Sequential(
                Upsamplefun(scale_factor=2, fun='Upsample', num_channels=num_channels_down, upsample_mode=upsample_mode),
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun)
            )
            self.conv = nn.Conv2d(2, 1, kernel_size = 1, stride = 1, padding = (1 - 1) // 2,
                                bias = False)
            self.zconv3 = nn.Sequential(
                conv(num_channels_down*2, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun))
            self.mode = 2
            
        elif mode == '1D':
            self.SA1 = nn.Sequential(
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun),
                )
            self.SA2 = nn.Sequential(
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun),
                )
            self.zconv21 = nn.Sequential(
                Upsamplefun(scale_factor=2, fun='Upsample', num_channels=num_channels_down, upsample_mode=upsample_mode),
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun)
            )
            self.zconv22 = nn.Sequential(
                Upsamplefun(scale_factor=2, fun='Upsample', num_channels=num_channels_down, upsample_mode=upsample_mode),
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun)
            )
            self.conv = nn.Conv1d(2, 1, kernel_size = 1, stride = 1, padding = (1 - 1) // 2,
                                bias = False)
            self.zconv3 = nn.Sequential(
                conv1d(num_channels_down*2, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun))
            self.mode = 1

        self.sigmoid = nn.Sigmoid()
        self.actf = act(act_fun)
        self.c1 = torch.mean
        self.c2 = torch.max
    def forward(self, Y1, Y2, Z):
        if self.mode == 2:
            Y1 = self.SA1(Y1)
            Y2 = self.SA2(Y2)

            Z11 = self.zconv21(Z)
            Z12 = self.zconv22(Z)
            Z2 = torch.mul(Crop(Z11, Y1), Crop(Y1, Z11))
            Z3 = torch.mul(Crop(Z12, Y2), Crop(Y2, Z12))
            Out = torch.cat([Crop(Z2, Z3), Crop(Z3, Z2)], dim=1)

            Out1 = self.zconv3(Out)
            avg_A = self.c1(Out1, dim = 1, keepdim = True)
            max_A, _ = self.c2(Out1, dim = 1, keepdim = True)
            v = self.conv(torch.cat((max_A, avg_A), dim = 1))
            Out = self.sigmoid(v) * Out1

        elif self.mode == 1:
            Y1 = self.SA1(Y1)
            Y2 = self.SA2(Y2)

            Z11 = self.zconv21(Z)
            Z12 = self.zconv22(Z)

            Z2 = torch.mul(Crop(Z11, Y1, mode='1D'), Crop(Y1, Z11, mode='1D'))
            Z3 = torch.mul(Crop(Z12, Y2, mode='1D'), Crop(Y2, Z12, mode='1D'))
            Out = torch.cat([Crop(Z2, Z3, mode='1D'), Crop(Z3, Z2, mode='1D')], dim=1)

            Out1 = self.zconv3(Out)
            avg_A = self.c1(Out1, dim = 1, keepdim = True)
            max_A, _ = self.c2(Out1, dim = 1, keepdim = True)
            v = self.conv(torch.cat((max_A, avg_A), dim = 1))
            Out = self.sigmoid(v) * Out1
        return Out




class DAU(nn.Module):
    def __init__(self, num_channels_down, mode='2D', act_fun='LeakyReLU'):
        super(DAU, self).__init__()
        if mode == '2D':
            self.SA = nn.Sequential(
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun))
            self.c1 = torch.mean
            self.c2 = torch.max
            self.conv = nn.Conv2d(2, 1, kernel_size = 1, stride = 1, padding = 0,
                                bias = False)
            self.zconv = nn.Sequential(
                conv(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn(num_channels_down),
                act(act_fun)
            )
            self.mode = 2
        elif mode == '1D':
            self.SA = nn.Sequential(
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun),)
            self.c1 = torch.mean
            self.c2 = torch.max
            self.conv = nn.Conv1d(2, 1, kernel_size = 1, stride = 1, padding = (1 - 1) // 2,
                                bias = False)
            self.zconv = nn.Sequential(
                conv1d(num_channels_down, num_channels_down, 1, bias=True, pad='reflection'),
                bn1d(num_channels_down),
                act(act_fun)
            )
            self.mode = 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, Y1, Z):
        AY = self.SA(Y1)
        Z1 = self.zconv(Z)
        if self.mode == 1:
            m = torch.mul(Crop(Z1, AY, mode='1D'), Crop(AY, Z1, mode='1D'))
        else:
            m = torch.mul(Crop(Z1, AY), Crop(AY, Z1))
        m = m + Z

        avg_A = self.c1(AY, dim = 1, keepdim = True)
        max_A, _ = self.c2(AY, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_A, avg_A), dim = 1))
        Out = self.sigmoid(v) * m + m
        return Out

class DCNBlock(nn.Module):
    def __init__(self, input_channel, output_channel, act_fun):
        super(DCNBlock, self).__init__()
        self.conv = nn.Sequential(
            DCN(input_channel, output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            bn(output_channel),
            act(act_fun)
        )
    def forward(self, x):
        Output = self.conv(x) + x
        return Output

def concat(input1, input2, mode='2D'):
    inputs = []
    inputs.append(input1)
    inputs.append(input2)
    if mode == '2D':
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        
        # equal dimension
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3: diff3 + target_shape3])
    elif mode == '1D':
        inputs_shapes2 = [x.shape[2] for x in inputs]
        
        # equal dimension
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2])
    return torch.cat(inputs_, dim=1)

def Crop(input1, input2, mode='2D'):
    # return input1
    inputs = []
    inputs.append(input1)
    inputs.append(input2)
    if mode == '2D':
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        
        # equal dimension
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = input1
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            # for inp in inputs:
            diff2 = (input1.size(2) - target_shape2) // 2
            diff3 = (input1.size(3) - target_shape3) // 2
            inputs_ = input1[:, :, diff2: diff2 + target_shape2, diff3: diff3 + target_shape3]
    elif mode == '1D':
        inputs_shapes2 = [x.shape[2] for x in inputs]
        
        # equal dimension
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)):
            inputs_ = input1
        else:
            target_shape2 = min(inputs_shapes2)
            diff2 = (input1.size(2) - target_shape2) // 2
            inputs_ = input1[:, :, diff2: diff2 + target_shape2]
    return inputs_



class GDIP2d(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3,  num_inputz_channels=20,
        num_channels=128, num_channels_skip=4, 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):

        super(GDIP2d, self).__init__()

        # y network
        self.yconv0 = nn.Sequential(
            conv(num_input_channels, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun)
        )

        self.decode = nn.Sequential(
            conv(num_channels, num_channels, filter_size_down, 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode),
            bn(num_channels),
            act(act_fun),
            conv(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun)
        )
        self.skip = nn.Sequential(
            conv(num_channels, num_channels_skip, 1, bias=True, pad='reflection'),
            bn(num_channels_skip),
            act(act_fun)
        )
        self.yencode0 = nn.Sequential(
            Upsamplefun(2, fun='Upsample', num_channels=num_channels + num_channels_skip, upsample_mode=upsample_mode),
            conv(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun),
        )
        self.yencode = nn.Sequential(
            Upsamplefun(2, fun='Upsample', num_channels=num_channels + num_channels_skip, upsample_mode=upsample_mode),
            conv(num_channels + num_channels_skip, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun)
        )
        self.youtput = nn.Sequential(
            conv(num_channels, num_output_channels, 1, bias=True, pad='reflection'),
            bn(num_output_channels),
            act(act_fun)
        )
        self.DCN11 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN12 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN13 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN14 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')

        # z network
        self.zconv0 = nn.Sequential(
            conv(num_inputz_channels, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun),
        )

        self.zconv = nn.Sequential(
            conv(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn(num_channels),
            act(act_fun))
        
        self.zoutput = nn.Sequential(
            conv(num_channels, num_output_channels, 1, bias=True, pad='reflection'),
            nn.ReLU(),
        )

        self.SAU = SAU(num_channels, act_fun='LeakyReLU', upsample_mode = 'bilinear')
        self.DCN21 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN22 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN23 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DCN24 = DCNBlock(num_channels, num_channels, act_fun='LeakyReLU')
        self.DAU = DAU(num_channels, act_fun='LeakyReLU')

        
    def forward(self, Y, Z):
        Y0 = self.yconv0(Y)
        Y1 = self.decode(Y0)
        Y2 = self.decode(Y1)
        Y3 = self.decode(Y2)
        Y4 = self.decode(Y3)
        Y5 = self.decode(Y4)
        skip1 = self.skip(Y1)
        skip2 = self.skip(Y2)
        skip3 = self.skip(Y3)
        skip4 = self.skip(Y4)


        Y01 = self.DCN11(Y5)
        Y02 = self.DCN12(Y01)
        Y03 = self.DCN13(Y02)
        Y04 = self.DCN14(Y03)

        Y6 = self.yencode0(Y04)
        Y7 = self.yencode(concat(Y6, skip4))
        Y8 = self.yencode(concat(Y7, skip3))
        Y9 = self.yencode(concat(Y8, skip2))
        Y10 = self.yencode(concat(Y9, skip1))

        Z0 = self.zconv0(Z)
        Z01 = self.DCN21(Z0)
        Z02 = self.DAU(Y01, Z01)

        Z03 = self.DCN22(Z02)
        Z04 = self.DAU(Y02, Z03)

        Z05 = self.DCN23(Z04)
        Z06 = self.DAU(Y03, Z05)

        Z07 = self.DCN24(Z06)
        Z08 = self.DAU(Y04, Z07)

        Z1 = self.SAU(Y4, Y6, Z08)

        Z1 = self.zconv(Z1) + Z1
        Z2 = self.SAU(Y3, Y7, Z1)
        Z2 = self.zconv(Z2) + Z2
        Z3 = self.SAU(Y2, Y8, Z2)
        Z3 = self.zconv(Z3) + Z3
        Z4 = self.SAU(Y1, Y9, Z3)
        Z4 = self.zconv(Z4) + Z4

        Z5 = self.SAU(Y0, Y10, Z4)
        Z5 = self.zconv(Z5) + Z5

        ZOut = self.zoutput(Z5)

        return ZOut



class GDIP1d(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_inputz_channels=20,
        num_channels=128, num_channels_skip=4, 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):

        super(GDIP1d, self).__init__()

        # y network
        self.yconv0 = nn.Sequential(
            conv1d(num_input_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.decode = nn.Sequential(
            conv1d(num_channels, num_channels, filter_size_down, 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode),
            bn1d(num_channels),
            act(act_fun),
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.skip = nn.Sequential(
            conv1d(num_channels, num_channels_skip, 1, bias=True, pad='reflection'),
            bn1d(num_channels_skip),
            act(act_fun),
        )
        self.yencode0 = nn.Sequential(
            Upsamplefun(2, fun='Upsample', mode='1D',num_channels=num_channels + num_channels_skip, upsample_mode=upsample_mode),
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.yencode = nn.Sequential(
            Upsamplefun(2, fun='Upsample', mode='1D', num_channels=num_channels + num_channels_skip, upsample_mode=upsample_mode),
            conv1d(num_channels + num_channels_skip, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.youtput = nn.Sequential(
            conv1d(num_channels, num_output_channels, 1, bias=True, pad='reflection'),
            bn1d(num_output_channels),
            act(act_fun)
        )
        self.DCN11 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN12 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN13 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN14 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )

        # z network
        self.zconv0 = nn.Sequential(
            conv1d(num_inputz_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun))

        self.zconv = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun))

        self.zoutput = nn.Sequential(
            conv1d(num_channels, num_output_channels, 1, bias=True, pad='reflection'),
            nn.ReLU(),
        )

        self.SAU = SAU(num_channels, mode='1D', act_fun='LeakyReLU', upsample_mode = 'linear')
        self.DCN21 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN22 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN23 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DCN24 = nn.Sequential(
            conv1d(num_channels, num_channels, 1, bias=True, pad='reflection'),
            bn1d(num_channels),
            act(act_fun)
        )
        self.DAU = DAU(num_channels, mode='1D', act_fun='LeakyReLU')

        
    def forward(self, Y, Z):
        Y0 = self.yconv0(Y)
        Y1 = self.decode(Y0)
        Y2 = self.decode(Y1)
        Y3 = self.decode(Y2)

        skip1 = self.skip(Y1)
        skip2 = self.skip(Y2)

        Y01 = self.DCN11(Y3) + Y3
        Y02 = self.DCN12(Y01) + Y01
        Y03 = self.DCN13(Y02) + Y02
        Y04 = self.DCN14(Y03) + Y03

        Y6 = self.yencode0(Y04)
        Y7 = self.yencode(concat(Y6, skip2, mode='1D'))
        Y8 = self.yencode(concat(Y7, skip1, mode='1D'))

        Z0 = self.zconv0(Z)
        Z01 = self.DCN21(Z0) + Z0
        Z02 = self.DAU(Y01, Z01)

        Z03 = self.DCN22(Z02) + Z02
        Z04 = self.DAU(Y02, Z03)

        Z05 = self.DCN23(Z04) + Z04
        Z06 = self.DAU(Y03, Z05)

        Z07 = self.DCN24(Z06) + Z06
        Z08 = self.DAU(Y04, Z07)

        Z1 = self.SAU(Y2, Y6, Z08)
        Z1 = self.zconv(Z1) + Z1
        Z2 = self.SAU(Y1, Y7, Z1)
        Z2 = self.zconv(Z2) + Z2
        Z3 = self.SAU(Y0, Y8, Z2)
        Z3 = self.zconv(Z3) + Z3

        ZOut = self.zoutput(Z3)

        return ZOut


class CGDIP(nn.Module):
    def __init__(self, num_input_channels1d=1024, num_inputz_channels1d=1024, num_output_channels=7,
                 num_input_channels2d=24, num_inputz_channels2d=24,
                 data_shape = [1,2,3],
        num_channels=128, num_channels_skip=4):

        super(CGDIP, self).__init__()
        self.data_shape = data_shape
        self.num_output_channels = num_output_channels
        self.net1d = GDIP1d(num_input_channels=num_input_channels1d, num_inputz_channels=num_inputz_channels1d, 
                            num_output_channels=num_output_channels, num_channels=num_channels, num_channels_skip=num_channels_skip)
        self.net2d = GDIP2d(num_input_channels=num_input_channels2d, num_inputz_channels=num_inputz_channels2d, 
                            num_output_channels=num_output_channels, num_channels=num_channels, num_channels_skip=num_channels_skip)
        self.relu = nn.ReLU()
        
    def forward(self, X_temp, G_temp1d, Y_temp, G_temp2d):
        U = [1,2]
        U[0] = self.net1d(X_temp, G_temp1d).reshape(self.num_output_channels, self.data_shape[0])
        U[1] = self.net2d(Y_temp, G_temp2d).reshape(self.num_output_channels, self.data_shape[1], self.data_shape[2])

        U[1] = torch.div(U[1], torch.sum(U[1], 0)+1e-7)
        UZ = [1,2]
        UZ[0] = U[0].clone().detach()
        UZ[1] = U[1].clone().detach()
        Z = torch.matmul(U[1].permute((1, 2, 0)), U[0]).permute((2, 0, 1))
        return Z, UZ, U
