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
conda create -n detr3d python=3.8 -y
conda activate detr3d
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



# 下载 detr3d
```
git clone https://github.com/WangYueFt/detr3d.git

```

# 安装numpy、yapf、filelock来避免一些bug
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



# 下载 mmdetection3d v1.0.0rc6
方法1. detr3d中的mmdetection3d模块安装（推荐）
detr3d代码中包含了mmdetection3d的子仓库，我们可以直接通过以下命令，拉取子模块的代码，注意要切换到v1.0.0rc6版本。安装mmdetection3d需要在后面一些库安装完之后再装

```
cd detr3d
git submodule init
git submodule update
cd mmdetection3d
git checkout -b v1.0.0rc6 v1.0.0rc6
```
方法2. 外部下载mmdetection3d
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git tag
git checkout -b v1.0.0rc6 v1.0.0rc6
```

# 安装 mmdetection3d v1.0.0rc6
前置所有依赖库安装完毕后，安装mmdetection3d。请确保已经checkout到了v1.0.0rc6
```
cd mmdetection3d
pip install -v -e .
```

如果报错说mmcv版本过高，我们可以修改mmdetection3d的__init__.py文件，将mmcv的版本检查去掉
mmdet/__init__.py
line 16
```
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version))
```

# 降低setuptools版本
之前的安装可能会把setuptools的版本提升到60.2.0，甚至75.0，版本过高，会在后面运行detr3d时报错，因此需要降级到59.5.0
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
cd detr3d
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

# 单GPU训练

1. 命令行模式
命令行的训练是通过tools/dist_train.sh进行

https://blog.csdn.net/XCCCCZ/article/details/134295931
在进行多卡训练时，首先我们要对nuscene的部分代码进行修改，否则会出现错误
打开conda环境中的对应文件
/home/ubuntu/anaconda3/envs/detr3d/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py
line 39
```
# self.class_names = self.class_range.keys()
self.class_names = list(self.class_range.keys())
```


```
tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py 1
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
            "args": ["projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py"],
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


# 多卡训练
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
tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 2
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
                "projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py",
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
参考https://github.com/WangYueFt/detr3d中Evaluation部分，下载对应的pth文件，放置在ckpt文件夹下

方法1. 命令行模式
```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/ckpt 1 --eval=bbox
```
方法2. vscode launch.json模式
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
                "projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py",
                "ckpt/detr3d_resnet101.pth",
                "--eval=bbox"
                ],
            ],
            "justMyCode": false

        }
    ]
}

```

# 多GPU测试

方法1. 命令行模式
```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/ckpt 3 --eval=bbox
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
                "projects/configs/detr3d/detr3d_res101_gridmask.py",
                "ckpt/detr3d_resnet101.pth",
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

# 可视化



# 在nuscenes数据集上进行测试

https://blog.csdn.net/Furtherisxgi/article/details/130118952
```
bash ./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth 1 --eval bbox

```

# Test of BevFusion on nuscenes
https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x/projects/BEVFusion
```
python projects/BEVFusion/setup.py develop

python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py checkpoints/bevfusion_converted.pth --cam-type all --score-thr 0.2 --show


python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth --cam-type all --score-thr 0.2 --show
```
