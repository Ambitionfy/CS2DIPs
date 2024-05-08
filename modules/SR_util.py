import torch
import numpy as np
import pypher
import cv2


def para_setting(kernel_type,sf,sz,delta):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, delta), (cv2.getGaussianKernel(sf, delta)).T)
    fft_B = pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


def H_z(z, factor, fft_B ):
    #     z  [31 , 96 , 96]
    #     ch, h, w = z.shape
    # f = torch.rfft(z, 2, onesided=False)
    f = torch.fft.fft2(z, dim=(-2, -1))
    # Hz = torch.zeros_like(f)
    f = torch.stack((f.real, f.imag), -1)
    # -------------------complex myltiply-----------------#
    if len(z.shape)==3:
        ch , h, w = z.shape
        fft_B = fft_B.unsqueeze(0).repeat(ch,1,1,1)
        M = torch.cat(( (f[:,:,:,0]*fft_B[:,:,:,0]-f[:,:,:,1]*fft_B[:,:,:,1]).unsqueeze(3) ,
                        (f[:,:,:,0]*fft_B[:,:,:,1]+f[:,:,:,1]*fft_B[:,:,:,0]).unsqueeze(3) )  ,3)
        # Hz = torch.irfft(M, 2, onesided=False)
        Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1))
        x = Hz[:, int(factor//2)::factor ,int(factor//2)::factor].real
    elif len(z.shape)==4:
        bs,ch,h,w = z.shape
        fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs,ch, 1, 1, 1)
        M = torch.cat((  (f[:,:,:,:,0]*fft_B[:,:,:,:,0]-f[:,:,:,:,1]*fft_B[:,:,:,:,1]).unsqueeze(4) ,
                         (f[:,:,:,:,0]*fft_B[:,:,:,:,1]+f[:,:,:,:,1]*fft_B[:,:,:,:,0]).unsqueeze(4) ), 4)
        Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1))
        x = Hz[: ,: , int(factor//2)::factor ,int(factor//2)::factor].real
    return x
