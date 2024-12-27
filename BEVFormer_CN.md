[TOC] 
参考了以下网页
https://blog.csdn.net/newbie_dqt/article/details/134766294

# 安装nvidia驱动
如果有旧的驱动，或者进不了桌面，先卸载旧的驱动
```
cd /usr/bin/
sudo ./nvidia-installer --uninstall
```
进入终端并且切换到init 模式安装，避免出现xserver的警告
```
ctr+alt+F3
sudo init 3
sudo ./NVIDIA-Linux-x86_64-550.90.07.run
```
切记，安装后有两个选择,32 compatibility和auto的，都选No


# 安装 gcc 8 and g++ 8
参考了以下网页
https://askubuntu.com/questions/1446863/trying-to-install-gcc-8-and-g-8-on-ubuntu-22-04
```
sudo apt update
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb
sudo apt install ./libisl22_0.22.1-1_amd64.deb ./libmpx2_8.4.0-3ubuntu2_amd64.deb ./cpp-8_8.4.0-3ubuntu2_amd64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb ./gcc-8_8.4.0-3ubuntu2_amd64.deb
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100

sudo ln -s /usr/bin/gcc /usr/bin/cc

wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb
sudo apt install ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb ./g++-8_8.4.0-3ubuntu2_amd64.deb

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 100

```

# 安装 CUDA 11.3
要注意安装的CUDA版本要和之后安装的Pytorch版本一致
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

# 创建conda环境
```
conda create -n BEVFormer python=3.8 -y
conda activate BEVFormer
```

# 安装 Pytroch 1.10.0
切记，用pip的安装法 这里面的cuda版本要是11.3，一定要和上面安装的CUDA版本一致
```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

```

安装完之后通过
```
pip list | grep torch

```
torch                                1.10.0+cu113
torchaudio                           0.10.0+cu113
torchvision                          0.11.0+cu113

确认安装的torch版本是1.10.0


# 下载 BEVFormer
```
git clone https://github.com/fundamentalvision/BEVFormer.git
```

# 安装 mmcv v1.4.0

```
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

# 安装 mmdet 2.14.0
```
pip install mmdet==2.14.0
```
# 安装 mmsegmentation 0.14.1
```
pip install mmsegmentation==0.14.1
```

# 安装一些代码中用到的依赖包（特别是nuscenes-devkit，官方步骤没有提到，但是代码确实用到了）
```
pip install filelock
pip install ninja 
pip install tensorboard==2.13.0 
pip install nuscenes-devkit==1.1.10 
pip install scikit-image==0.19.0 
pip install lyft-dataset-sdk==0.0.8

```

# 下载BEVFormer
```
git clone https://github.com/fundamentalvision/BEVFormer.git
```
# 下载并安装mmdetection3d v0.17.1
```
cd BEVFormer
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v e .
```
# 安装一些三方库
```
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12 numba==0.48.0 scikit-image==0.19.3 yapf==0.40.1 
```

# 改变numpy、pandas、setuptools版本，安装llvmlite
```
pip install numpy==1.19.5 
pip install pandas==1.4.4 
pip install llvmlite==0.31.0 
pip install setuptools==59.5.0
```

# 安装 Detectron2，网卡会中断可以选择手动下载手动安装或者多试几次
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```
# 处理 nuscenes data
当我们用nuscenes数据进行训练时，需要先下载nuscenes数据集，然后进行预处理，生成训练用的数据集
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html

注意：由于bevformer安装了Detectron2，因此会有一个叫tools的工具安装到了python环境中，会抢tools的文件夹，
因此tools.data_converter/indoor_converter.py中
```
from tools.data_converter.s3dis_data_utils import S3DISData, S3DISSegData
from tools.data_converter.scannet_data_utils import ScanNetData, ScanNetSegData
from tools.data_converter.sunrgbd_data_utils import SUNRGBDData
```
要改成
```
from data_converter.s3dis_data_utils import S3DISData, S3DISSegData
from data_converter.scannet_data_utils import ScanNetData, ScanNetSegData
from data_converter.sunrgbd_data_utils import SUNRGBDData
```
```
cd MapTR
mkdir data
cd data
ln -s /path/to/nuscenes ./
ln -s /path/to/can_bus ./
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

# 处理 nuscenes mini data
如果想用小一点的nuscenes mini数据集进行训练，可以进行如下操作
1. 下载nuscenes mini数据集，解压到v1.0-mini文件夹中
2. 在data文件夹下把v1.0-mini软连接过来重命名为nuscenes-mini
```
cd data
ln -s /path/to/nuscenes-mini nuscenes-mini
ln -s /path/to/can_bus ./
```
3. 对mini数据集进行预处理
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes-mini --out-dir ./data/nuscenes-mini --extra-tag nuscenes --version v1.0-mini --canbus ./data

```

# 下载预训练的backbone
```
cd bevformer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```



# 单GPU训练

1. 命令行模式
命令行的训练是通过tools/dist_train.sh进行
```
bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 1
```
其中1代表单卡
如果遇到显存不足得问题，可以改为bevformer_small.py
```
bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_small.py 1
```

2. vscode的launch.json配置模式
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "tools/train.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": ["projects/configs/bevformer/bevformer_small.py"],
            // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"],
            "justMyCode": false

        }
    ]
}

```

注意，有肯能会出现一个CUDA相关的错误
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`，
```
这里参照
https://blog.csdn.net/BetrayFree/article/details/133868929的说法,可以通过
```
unset LD_LIBRARY_PATH
```
来解决

实际上很可能是因为多版本cuda的问题，所以根本性的解决方式是应该卸载所有cuda，然后之安装11.3


# 多GPU训练
https://blog.csdn.net/XCCCCZ/article/details/134295931
在进行多卡训练时，首先我们要对nuscene的部分代码进行修改，否则会出现错误
打开conda环境中的对应文件
/home/ubuntu/anaconda3/envs/detr3d/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py
line 39
```
# self.class_names = self.class_range.keys()
self.class_names = list(self.class_range.keys())
```
方法1. 命令行模式
Then Run the training code
```
bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 3
```

如果遇到显存不足得问题，可以改为bevformer_small.py
```
bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_small.py 3
```

方法2. vscode launch.json模式
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            // "program": "ddp_test.py",
            "console": "integratedTerminal",
            "module": "torch.distributed.run",
            "args": [
                "--nproc_per_node", "3",
                "tools/train.py",
                "--launcher=pytorch",
                "projects/configs/bevformer/bevformer_small.py",
                // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"
            ],
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}

```

# 单GPU测试
测试时需要下载对应的pth文件
参考https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file，下载对应的pth文件，放置在ckpt文件夹下

方法1. 命令行模式
```
tools/dist_test.sh projects/configs/bevformer/bevformer_small.py pretrained/bevformer_small_epoch_24.pth 1
```
方法2. vscode launch.json模式
注意需要修改tools/test.py中代码
```
    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
```
变为
```
    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
```

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "tools/test.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": [
                "projects/configs/bevformer/bevformer_small.py",
                "pretrained/bevformer_small_epoch_24.pth",
                "--eval=box"
                ],
            "justMyCode": false
        }
    ]
}

```

# 多GPU测试

方法1. 命令行模式
```
tools/dist_test.sh projects/configs/bevformer/bevformer_small.py pretrained/bevformer_small_epoch_24.pth 3
```

方法2. vscode launch.json模式
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            // "program": "ddp_test.py",
            "console": "integratedTerminal",
            "module": "torch.distributed.run",
            "args": [
                "--nproc_per_node", "3",
                "tools/test.py",
                "--launcher=pytorch",
                "projects/configs/bevformer/bevformer_small.py",
                "pretrained/bevformer_small_epoch_24.pth",
                "--eval=box"
            ],
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}

```

# 可视化
首先要通过
```
tools/dist_test.sh projects/configs/bevformer/bevformer_small.py pretrained/bevformer_small_epoch_24.pth 1
```
产生检测结果，结果保存在test文件夹下，例如
```
test/bevformer_small/Thu_Dec_26_15_02_00_2024/pts_bbox/results_nusc.json
```
注意要修改tools/analysis_tools/visual.py中的
```
bevformer_results = mmcv.load('test/bevformer_small/Thu_Dec_26_15_02_00_2024/pts_bbox/results_nusc.json')
```
load的路径改为产生的json

然后运行
```
python tools/analysis_tools/visual.py
```
生成图。如果跳出来摄像头的图不动了，关闭这个图，他就会继续运行
