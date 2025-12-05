import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    return data/255.

#分割函数：将原图片分割成数个小图像块
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

#数据处理函数：训练数据——多尺度、多增强的小块  测试数据——完整图像
def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        # 为避免在 DataLoader 使用多进程（Windows 的 spawn 或其他情况）时
        # 将不可序列化的 h5py.File 对象随 Dataset 一起被 pickle 导致错误，
        # 我们改为：
        # - 存储 HDF5 文件名（`self.filename`）和 keys 列表；
        # - 不在 __init__ 中长期保存 h5py.File 引用；
        # - 在第一次需要读取时（或每个 worker 中第一次调用 __getitem__）再打开 h5 文件并缓存到 `self.h5f`。
        # 这样 Dataset 实例可以安全地被 pickle/传递给子进程，而无需序列化 h5py.File 对象。
        filename = 'train.h5' if self.train else 'val.h5'
        self.filename = filename
        # 仅短暂打开以读取 keys，然后立即关闭，保证 keys 可用于索引且不会留下不可序列化的句柄
        with h5py.File(filename, 'r') as _f:
            self.keys = list(_f.keys())
        random.shuffle(self.keys)
        # 延迟打开（按需）
        self.h5f = None
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        # 直接从已打开的 HDF5 文件中读取数据并返回张量
        # 按需打开 HDF5 文件（如果尚未打开）——这避免了在 pickle 时包含 h5py.File 对象
        if self.h5f is None:
            # 打开为只读，且保持在该 Dataset 实例生命周期内
            self.h5f = h5py.File(self.filename, 'r')
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)

    def __del__(self):
        # 在对象析构时关闭 HDF5 文件，防止文件句柄泄露。
        # 如果 Dataset 在 worker 中被多次创建/销毁，这里会在各自作用域中关闭对应句柄。
        if hasattr(self, 'h5f') and self.h5f is not None:
            try:
                self.h5f.close()
            except Exception:
                # 避免析构时抛出异常影响程序退出
                pass

    def __getstate__(self):
        # 在 pickle 时排除不可序列化的 h5py.File 对象
        state = self.__dict__.copy()
        state['h5f'] = None
        return state
