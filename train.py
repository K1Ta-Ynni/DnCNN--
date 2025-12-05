import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp  # 自动混合精度（Automatic Mixed Precision）
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

#模型训练主函数
def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    # 使用 pin_memory 可以在 host->device 传输时略微加速（适用于 GPU 训练）
    # 使用多个 worker 已经安全：Dataset 已修改为按需打开 HDF5，避免 h5py 对象被 pickle。
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, pin_memory=True, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    #criterion = nn.MSELoss(size_average=False)  已过时
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # 混合精度（AMP）相关：使用 GradScaler 来缩放 loss 并安全地进行反向传播与优化步。
    # 要求 PyTorch >= 1.6，若使用较旧版本请改用 apex 或不使用 AMP。
    scaler = amp.GradScaler()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            optimizer.zero_grad()
            # 将整个 batch 提前移动到 GPU；配合 DataLoader 的 pin_memory 和 non_blocking=True 可提高拷贝效率
            img_train = data.cuda(non_blocking=True)
            # 在已在设备上的张量上生成噪声，避免在 CPU 和 GPU 之间拷贝
            if opt.mode == 'S':
                # 固定噪声等级：与原代码等价，但使用与 img 相同的 dtype/device
                noise = torch.randn_like(img_train) * (opt.noiseL/255.)
            elif opt.mode == 'B':
                # 盲噪声：为每个样本生成不同的标准差
                batch_size = img_train.size(0)
                stdN = torch.rand(batch_size, device=img_train.device) * (55/255.)
                noise = torch.randn_like(img_train) * stdN.view(-1,1,1,1)
            imgn_train = img_train + noise

            # 使用 AMP 自动混合精度来加速 forward 并减少显存，占位于 autocast 上下文中
            # 并通过 GradScaler 来安全处理梯度缩放、optimizer.step
            with amp.autocast():
                out_train = model(imgn_train)
                loss = criterion(out_train, noise) / (imgn_train.size(0) * 2)

            # scale -> backward -> step -> update 的标准 AMP 流程
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # results
            model.eval()
            with torch.no_grad():
                out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        #为了在验证阶段确保不计算梯度，并替代 volatile=True 的作用，在此次使用 torch.no_grad() 上下文管理器来包裹整个验证循环
        with torch.no_grad():
            # validate
            psnr_val = 0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                #img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)  已过时
                img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
                out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
