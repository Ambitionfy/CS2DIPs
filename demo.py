import scipy.io as sio
import numpy as np
import math
import argparse
import os
import torch
import sys
import tqdm
sys.path.append('modules')
import utils
import SR_util
import model

def create_P():
    P_ = sio.loadmat('data/R.mat')['R']
    P_ = P_[:, 0:102]
    P = np.array(P_, dtype=np.float32)
    div_t = np.sum(P, axis=1)
    for i in range(P.shape[0]):
        P[i,] = P[i,]/div_t[i]
    return P

def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='data/',        help='dataset dir.')
    parser.add_argument("--dataset_name", type=str, default='PaviaC.mat', help='test data name.')
    parser.add_argument("--label", type=str, default='data',                        help='data label.')
    parser.add_argument("--gpu", type=int, default=0,                       help='GPU Index.')
    parser.add_argument("--delta", type=float, default=3)
    parser.add_argument("--kernel", type=float, default=8)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--epoches", type=int, default=5000)
    parser.add_argument("--nettype", type=str, default='dip')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rank1", type=int, default=35)
    parser.add_argument("--rank2", type=int, default=10)
    parser.add_argument("--rank3", type=int, default=256)
    parser.add_argument("--reg_noise_std", type=float, default=1.0/30.0)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    args.cuda = not (args.gpu is None) and torch.cuda.is_available()
    print("Parameters: ")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print('\n')
    print('---------------------')
    print('Loading dataset...')
    data_dir = args.dataset_dir+args.dataset_name
    data = sio.loadmat(data_dir)[args.label].astype(np.float32).transpose([2,0,1])

    setup_seed(seed=766)
    data = utils.normalize(data)
    print(data.shape)
    l1 = torch.nn.HuberLoss(reduction='sum', delta=0.1)
    fft_B,fft_BT = SR_util.para_setting('gaussian_blur',args.kernel,[data.shape[1], data.shape[2]],delta=args.delta)
    fft_B = torch.cat( (torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)) ,2 ).cuda()
    fft_BT = torch.cat( (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)) ,2 ).cuda()

    data_tensor = torch.FloatTensor(data).cuda()

    P = torch.FloatTensor(create_P())
    P = P.cuda()
    X = SR_util.H_z(data_tensor, args.factor, fft_B).cuda()
    Y = torch.tensordot(P, data_tensor, dims=([1],[0])).cuda()

    Nzx, Mx, Nx = X.shape
    Nzy, My, Ny = Y.shape
    print(X.shape)
    print(Y.shape)


    # network
    if args.nettype == 'dip':
        G = []
        Net = []
        net_param = []
        inp_param = []
        G_clone = []
        G_temp = []
        U = []
        UZ = []
        tempy1 = math.ceil(My / 2**5)
        tempy2 = math.ceil(Ny / 2**5)
        tempx = math.ceil(Nzx / 2**3)
        
        Net.append(model.CGDIP(num_input_channels1d=Nx*Mx, num_inputz_channels1d=args.rank3, 
                                num_input_channels2d=Nzy, num_inputz_channels2d=args.rank2,
                                data_shape = [data.shape[0], data.shape[1], data.shape[2]],
                                num_output_channels=args.rank1, num_channels=256, num_channels_skip=8).cuda())
        Net[0].train()
        net_param = list(Net[0].parameters())
        for i in range(2):
            if i == 0:
                G.append(utils.get_inp([1, args.rank3, tempx]))
                X_temp = X.detach().unsqueeze(0).permute(0,2,3,1).reshape(1,Mx*Nx,Nzx).cuda()
                
            elif i == 1:
                G.append(utils.get_inp([1, args.rank2, tempy1, tempy2]))
                Y_temp = Y.detach().unsqueeze(0).cuda()

            G_clone.append(G[i].detach().clone())
            inp_param = inp_param + [G[i]]
            G_temp.append([])
            U.append([])
            UZ.append([])

        params = net_param + inp_param

        optimizer = torch.optim.Adam(lr=args.lr, params=params)

    best_psnr = 0
    best_mse = 10000
    tbar = tqdm.tqdm(range(args.epoches))
    pp =[]
    for idx in tbar:
        # theta-subproblem
        G_temp[0] = G[0] + G_clone[0].normal_() * args.reg_noise_std
        G_temp[1] = G[1] + G_clone[1].normal_() * args.reg_noise_std

        Z, UZ, U = Net[0](X_temp, G_temp[0], Y_temp, G_temp[1])

        total_loss = l1(Y, torch.tensordot(P, Z, dims=([1],[0])))
        temp = SR_util.H_z(Z, args.factor, fft_B)
        total_loss = total_loss + l1(X, temp)

        mse1 = total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        Z1 = Z.detach().cpu().numpy()
        psnr = utils.psnr_3(Z1, data, data_range=1.0)
        pp.append(psnr)
        tbar.set_description('%.4e | %.2f | %.2f | %.2f'%(total_loss, mse1, psnr, best_psnr))
        tbar.refresh()
        
        if psnr > best_psnr:
            best_psnr = psnr

        if mse1 < best_mse:
            bestz2 = Z1.copy()
            best_mse = mse1

    print('-----------------------')
    bestz2 = np.array(bestz2)

    mse= utils.compute_mse(bestz2, data)
    psnr3 = utils.psnr_3(bestz2, data, data_range = 1.0)
    print(psnr3)
    print(utils.compute_ssim(bestz2, data))
    print(utils.compute_sam(bestz2, data))
    print(utils.compute_ergas(mse, data,r = args.factor))

if __name__ == "__main__":
    main()