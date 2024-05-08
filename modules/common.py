import torch
import torch.nn as nn
import numpy as np


def add_module(self, module):
    self.add_module(str(len(self)), module)

torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

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
        return torch.cat(inputs_, dim=self.dim)

class Concat1d(nn.Module):
    def __init__(self, dim, *args):
        super(Concat1d, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)):
            inputs_ = inputs
        else:
            target_shapes2 = min(inputs_shapes2)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shapes2) // 2
                inputs_.append(inp[:, :, diff2 : diff2 + target_shapes2])
        
        return torch.cat(inputs_, dim=self.dim)



class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()
    def forward(self, x):
        return x * self.s(x)

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
        self.omega = 30
    def forward(self, x):
        return torch.sin(self.omega * x)


def bn(num_features):
    return nn.BatchNorm2d(num_features)

def bn1d(num_features):
    return nn.BatchNorm1d(num_features) 

def conv(in_channels, out_channels, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        # elif downsample_mode in ['lanczos2', 'lanczos3']:
        #     downsampler = Downsampler(n_)
        else:
            assert False
        stride = 1
    
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    
    convolver = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x:x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def conv1d(in_channels, out_channels, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool1d(stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool1d(stride)
        else:
            assert False
        stride = 1
    
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad1d(to_pad)
        to_pad = 0
    
    convolver = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x:x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'sine':
            return Sine()
        else:
            assert False
    else:
        return act_fun()