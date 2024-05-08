import torch
import numpy as np
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def normalize(x):
    return x / np.max(x)


def get_inp(tensorSize, const=10.0):
    # get a variale
    inp = torch.rand(tensorSize, requires_grad=True).cuda()/const
    inp = torch.nn.Parameter(inp)
    return inp


def psnr_3(img, imclean, data_range):
    img_cpu = img.astype(np.float32)
    imgclean = imclean.astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(np.clip(imgclean[i, :, :],0,1), np.clip(img_cpu[i, :, :],0,1), data_range=data_range)
    return psnr / img_cpu.shape[0]


def compute_mse(img1, img2):
    w,h,c = img1.shape
    img1 = np.reshape(img1, (w*h,c))
    img2 = np.reshape(img2, (w*h,c))
    mse_sum  = (img1  - img2 )**2
    mse_loss = mse_sum.mean(0)
    mse_loss = mse_loss
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    return mse_loss

def compute_sam(im1, im2):
    w,h,c = im1.shape
    im1 = np.reshape(im1,(w*h,c))
    im2 = np.reshape(im2,(w*h,c))
    mole = np.sum(np.multiply(im1, im2), axis=1) 
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole+10e-8)/(deno+10e-8)).clip(-1,1)))
    return np.mean(sam)

def compute_ssim(im1,im2): 
    w,h,c = im1.shape
    im1 = np.reshape(im1,(w,h,c))
    im2 = np.reshape(im2,(w,h,c))
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = sk_cpt_ssim(im1[:,:,i], im2[:,:,i])
        ms_ssim += single_ssim
    return ms_ssim/n

def compute_ergas(mse, out, r=8):
    w,h,c = out.shape
    out = np.reshape(out,(w*h,c))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (c, 1))
    out_mean = np.reshape(out_mean, (c, 1))
    ergas = 100/r*np.sqrt(np.mean(mse/out_mean**2))                    
    return ergas