# DnCNN-PyTorch NeiKos496小组作业

这是 TIP2017 论文 Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising 的 PyTorch 实现(http://ieeexplore.ieee.org/document/7839189/)。

作者的 MATLAB 实现请见此处(https://github.com/cszn/DnCNN).

原代码是使用 PyTorch < 0.4 编写的，本次作业对低版本语法进行了优化使得适应 PyTorch 2.5.1

如何运行

1. 依赖项 
PyTorch(2.5.1)
torchvision
OpenCV for Python
HDF5 for Python
tensorboardX (PyTorch 的 TensorBoard 可视化工具)

2.训练 DnCNN-S/B (已知噪声水平的 DnCNN)

conda activate myenv
cd /home/xie/zzzmypython/DnCNNpytorch

第一次加上预训练
--preprocess True 

后续训练：
nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 15 --val_noiseL 15 --outf logs/DnCNN-S-15 > logs/DnCNN-S-15.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --outf logs/DnCNN-S-25  > logs/DnCNN-S-25.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 50 --val_noiseL 50 --outf logs/DnCNN-S-50 > logs/DnCNN-S-50.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 20 --mode B --val_noiseL 25 --outf logs/DnCNN-B  > logs/DnCNN-B.log 2>&1 &

测试set68
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set68" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 50

### BSD68 平均 RSNR（最后两个模型为本次作业复现）
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-pytorch | DnCNN-B-pytorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |      31.70      |      30.90      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |      29.17      |      28.34      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |      26.16      |      25.68      |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|


测试set12
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set12" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 50

### Set12 平均 PSNR
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-pytorch | DnCNN-B-pytorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      | 32.859  | 32.680  |     32.837      |     32.725      |     32.811      |     31.810      |
|     25      | 30.436  | 30.362  |     30.404      |     30.344      |     30.349      |     29.219      |
|     50      | 27.178  | 27.206  |     27.165      |     27.138      |     27.057      |     26.435      |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|


tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-15
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-25
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-50
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-B

## 数据加载与训练优化说明

- **HDF5 按需打开（已更新）**: 为解决 Windows 上当 `DataLoader` 使用多进程时出现的 `TypeError: h5py objects cannot be pickled` 问题，`dataset.py` 中的 `Dataset` 类已改为按需打开 HDF5：
	- 在 `__init__` 中只保存 HDF5 文件名 `self.filename`，并短暂打开文件以读取 keys（随后立即关闭）。
	- 在第一次调用 `__getitem__` 时再打开 `h5py.File(self.filename, 'r')` 并缓存到 `self.h5f`，避免把 `h5py.File` 对象包含进 pickle 流程。
	- 实现了 `__getstate__`（将 `h5f` 在序列化时置为 `None`），并在 `__del__` 中关闭文件句柄，确保资源不会泄露。
	- 优点：`Dataset` 现在可以安全地被传给 `DataLoader` 的子进程（Windows 的 spawn 或其他需要 pickle 的上下文），同时保留每个 worker 各自打开文件以避免竞态。

- **`num_workers` 已恢复为多进程**: 基于上述改动，`train.py` 中已把 `DataLoader(..., num_workers=4, ...)` 恢复为默认多线程（可按需调节）。

- **混合精度训练（AMP）**: `train.py` 已集成 `torch.cuda.amp`：
	- 使用 `with amp.autocast():` 包裹前向计算以启用自动混合精度；使用 `amp.GradScaler()` 缩放 loss 并通过 `scaler.scale(loss).backward()`、`scaler.step(optimizer)`、`scaler.update()` 完成反向与优化步骤。
	- 要求 PyTorch 版本 >= 1.6（建议与当前 README 中列出的 PyTorch 版本保持一致）。如果你的环境不支持 `torch.cuda.amp`，请改用 apex 或禁用 AMP。

- **设备拷贝优化**: 已将 batch 提前移动到 GPU（`data.cuda(non_blocking=True)`）并在 `DataLoader` 中启用 `pin_memory=True`，以减少 host→device 传输延迟。

- **盲噪声生成优化**: 在盲噪声（mode=B）模式下，改为在 GPU 上为每个样本生成不同标准差的噪声（`stdN`），避免了 CPU↔GPU 的频繁拷贝。

- **快速检查命令（Windows PowerShell）**: 在项目目录（包含 `train.py` 的目录）下运行一个短的 dry-run 以验证改动：
```powershell
python train.py --preprocess False --batchSize 32 --epochs 1
```

- **注意与后续改进建议**:
	- 若在 Linux/macOS 使用 `fork` 启动方式并启用大量 `num_workers` 时仍遇到 HDF5 并发问题，可改为在 worker 初始化（`worker_init_fn`）中打开 HDF5 文件，或为每个 worker 在第一次访问时打开并缓存文件（当前已实现）。
	- 若计划扩展到多卡多机训练，考虑使用 `DistributedDataParallel (DDP)` 替代 `DataParallel`。
	- 根据你的 CPU 与 I/O 能力调整 `DataLoader` 的 `num_workers`（Windows 上 2-4 为常见起点）。