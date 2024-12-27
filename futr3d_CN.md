[TOC] 
参考了以下网页
https://blog.csdn.net/weixin_42545475/article/details/132422665

https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html

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
conda create -n futr3d python=3.8 -y
conda activate futr3d
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



# 下载 futr3d
```
git clone https://github.com/WangYueFt/detr3d.git

```

# 安装numpy和yapf来避免一些bug
参考以下网页，需要提前安装numpy和yapf的特定版本，否则回报错
https://github.com/open-mmlab/mmdetection/issues/10962
```
pip install numpy==1.23.5
pip install yapf==0.40.1
pip install filelock

```

# 下载安装 mmseg 0.30.0
方法1.通过源代码安装
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git tag
git checkout -b v0.30.0 v0.30.0
pip install -v -e .
```
方法2. 通过pip 安装（推荐）
```
pip install mmsegmentation==0.30.0
```

# 下载安装 mmdetection 2.28.0
方法1.通过源代码安装
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git tag
git checkout -b v2.28.0 v2.28.0
pip install -v -e .
```
方法2.通过mim安装
```
pip install -U openmim
mim install mmdet==2.28.0
```

方法3. 通过pip安装(推荐)
```
pip install mmdet==2.28.0
```

# 下载安装 mmengine 0.7.1
方法1.通过源代码安装
```
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
git tag
git checkout -b v0.7.1 v0.7.1
pip install -v -e .
```
方法2.通过mim安装
```
pip install -U openmim
mim install mmengine==0.7.1
```

方法3. 通过pip安装（推荐）
```
pip install mmengine==0.7.1
```

# 安装 mmcv v1.6.0
安装mmcv一定要和cuda和torch版本对应，否则之后会报错，请注意链接里面cu和torch的版本，链接可以尝试打开，看是否有1.6.0的mmcv
```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

如果出现mmcv/_ext.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPvmm
这种错误，基本就是mmcv和cuda版本不匹配，需要重新调整mmcv的版本


# 编译futr3d
futr3d本身就是直接在mmdetection3d的目录里的，因此不需要单独下载mmdtection3d，可以直接编译
```
cd futr3d
pip3 install -v -e .

```


# 降低setuptools版本
之前如果用openmim安装库，可能会把setuptools的版本提升到60.2.0，版本过高，会在后面运行detr3d时报错，因此需要降级到59.5.0
首先通过
```
pip list | grep setuptools
```
确定当前setuptools版本

```
pip install setuptools==59.5.0
```

# 处理 nuscenes data
当我们用nuscenes数据进行训练时，需要先下载nuscenes数据集，然后进行预处理，生成训练用的数据集
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html

```
cd futr3d
mkdir data
cd data
ln -s /path/to/nuscenes ./
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
注意要把create_data.py中的第224行注释掉，把代码放出来，否则代码不会生成train_val部分
```
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
```
运行完之后nuscenes文件夹应该是如下结构
~/Downloads/test_detr3d/detr3d/data/nuscenes$ tree -L 1
.
├── maps
├── nuscenes_dbinfos_train.pkl
├── nuscenes_gt_database
├── nuscenes_infos_test_mono3d.coco.json
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train_mono3d.coco.json
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val_mono3d.coco.json
├── nuscenes_infos_val.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval

# 处理 nuscenes mini data
如果想用小一点的nuscenes mini数据集进行训练，可以进行如下操作
1. 下载nuscenes mini数据集，解压到v1.0-mini文件夹中
2. 在data文件夹下把v1.0-mini软连接过来重命名为nuscenes-mini
```
cd data
ln -s ~/Downloads/v1.0-mini nuscenes-mini
```
3. 对mini数据集进行预处理
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes-mini --out-dir ./data/nuscenes-mini --extra-tag nuscenes --version v1.0-mini

```

# 下载预训练的backbone
https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN
我们可以下载fcos3d.pth作为预训练模型，放置在detr3d/pretrained目录下

# 修改futr3d代码以避免bug
1. 命令行符号问题
命令行的训练和测试是通过tools/dist_train.sh以及tools/dist_test.sh进行。futr3d的这两个的文件格式有问题，他是在windows下写成的，因此需要
```
sudo apt install dos2unix
dos2unix tools/dist_train.sh
dos2unix tools/dist_test.sh
```
2. lidar_0075v_900q.py内路径问题
plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py配置文件头部的引用层级不对，需要修改

```
_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/schedules/cyclic_20e.py', 
    '../../../configs/_base_/default_runtime.py'
]
```

变成
```
_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/schedules/cyclic_20e.py', 
    '../../../../configs/_base_/default_runtime.py'
]
```

3. No module named 'plugin.fudet'
代码中写错了库名，需要把代码中所有的plugin.fudet改成plugin.futr3d，包括三处
plugin/futr3d/datasets/loading.py
plugin/futr3d/models/detectors/futr3d.py
plugin/futr3d/models/head/futr3d_head.py

4. 显存不足
   如果训练时遇到
   ```
   RuntimeError: CUDA out of memory. Tried to allocate 170.00 MiB
   ```
   显存不足的问题，需要减少batch
   修改configs/_base_/datasets/nus-3d.py
```
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
```
修改为
```
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
```
如果还不够就变成samples_per_gpu=1


# 单GPU训练

1. 命令行模式

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py 1
```
其中1代表单卡

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
            "args": ["plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py"],
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
在进行多卡训练时，可能会有一个和nuscenes相关的error
https://blog.csdn.net/XCCCCZ/article/details/134295931
如果出现，需要进行修改
打开conda环境中的对应文件
/home/ubuntu/anaconda3/envs/detr3d/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py
line 39
```
# self.class_names = self.class_range.keys()
self.class_names = list(self.class_range.keys())
```

方法1. 命令行模式
```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py 3
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
                "plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py",
                // "--resume-from","./work_dirs/lidar_0075v_900q/latest.pth"
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
参考https://github.com/WangYueFt/detr3d中Evaluation部分，下载对应的pth文件，放置在ckpt文件夹下

方法1. 命令行模式
```
bash tools/dist_test.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py ckpt/lidar_0075_cam_res101_900q.pth 1 --eval bbox
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
            "program": "tools/test.py",
            "console": "integratedTerminal",
            "args": [
                "plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py",
                "ckpt/lidar_0075_cam_res101_900q.pth",
                "--eval=box"
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


# 多GPU测试
方法1. 命令行模式 
```
bash tools/dist_test.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py ckpt/lidar_0075_cam_res101_900q.pth 3 --eval bbox
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
                "plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py",
                "ckpt/lidar_0075_cam_res101_900q.pth",
                "--eval=box"
                // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"
            ],
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}
