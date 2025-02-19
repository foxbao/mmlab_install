# BEVDet 配置指南

[TOC]

参考了以下网页
<https://blog.csdn.net/weixin_42545475/article/details/132422665>

<https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html>

## 安装nvidia驱动


# 安装 gcc 12 and g++ 12
参考了以下网页
https://blog.csdn.net/qq_42059060/article/details/135677793
```
sudo apt-get install gcc-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
```

# 安装nvidia驱动
如果有旧的驱动，或者进不了桌面，先卸载旧的驱动

```bash
cd /usr/bin/
sudo ./nvidia-installer --uninstall
```

进入终端并且切换到init 模式安装，避免出现xserver的警告

```bash
ctr+alt+F3
sudo init 3
sudo ./NVIDIA-Linux-x86_64-550.90.07.run
```

切记，安装后有两个选择,32 compatibility和auto的，都选No

## 安装 gcc 8 and g++ 8



## 安装 CUDA 11.3

要注意安装的CUDA版本要和之后安装的Pytorch版本一致

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

## 创建conda环境

```bash
conda create -n BEVDet python=3.8 -y
conda activate BEVDet
```

## 安装 Pytroch 1.10.0

切记，用pip的安装法 这里面的cuda版本要是11.3，一定要和上面安装的CUDA版本一致

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

安装完之后通过

```bash
pip list | grep torch
```

torch                                1.10.0+cu113
torchaudio                           0.10.0+cu113
torchvision                          0.11.0+cu113
确认安装的torch版本是1.10.0

## 下载 BEVDet

```bash
git clone https://github.com/HuangJunJie2017/BEVDet.git
git checkout
```

## 安装mmcv 1.5.3

```bash
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

## 安装onnxruntime-gpu 1.8.1

从<https://dashboard.stablebuild.com/pypi-deleted-packages/pkg/onnxruntime-gpu/1.8.1下载>
onnxruntime_gpu-1.8.1-cp38-cp38-manylinux2014_x86_64.whl

```bash
pip install onnxruntime_gpu-1.8.1-cp38-cp38-manylinux2014_x86_64.whl
```

## 安装mmdet 2.25.1

```bash
pip install mmdet==2.25.1
```

## 安装mmsegmentation 0.25.0

```bash
pip install mmsegmentation==0.25.0
```

## 进入BEVDet工程目录,安装mmdet3d

```bash
cd BEVDet
pip install -v -e .
```

## 安装其他依赖 numpy==1.23.4 setuptools==58.2.0等

```bash
pip install pycuda 
pip install lyft_dataset_sdk 
pip install networkx==2.2 
pip install numba==0.53.0 
pip install numpy==1.23.4 
pip install nuscenes-devkit 
pip install plyfile 
pip install scikit-image 
pip install tensorboard 
pip install trimesh==2.35.39 
pip install setuptools==58.2.0 
pip install yapf==0.40.1
pip install spconv-cu113
```

## 处理 nuscenes data

当我们用nuscenes数据进行训练时，需要先下载nuscenes数据集，然后进行预处理，生成训练用的数据集
<https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html>

```bash
cd BEVDet
mkdir data
cd data
ln -s /path/to/nuscenes ./
python tools/create_data_bevdet.py
```

## 处理 nuscenes mini data

如果想用小一点的nuscenes mini数据集进行训练，可以进行如下操作

1. 下载nuscenes mini数据集，解压到v1.0-mini文件夹中
2. 在data文件夹下把v1.0-mini软连接过来重命名为nuscenes-mini

```bash
cd data
ln -s ~/Downloads/v1.0-mini nuscenes-mini
```

3. 对mini数据集进行预处理

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes-mini --out-dir ./data/nuscenes-mini --extra-tag nuscenes --version v1.0-mini
```

## 下载预训练模型

```bash
# 1 新建ckpts文件夹并进入
mkdir ckpts && cd ckpts

# 2 下载resnet50-0676ba61.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
```

## 单GPU训练

1. 命令行模式

    ```bash
    python tools/train.py ./configs/bevdet/bevdet-r50.py
    ```

2. vscode的launch.json配置模式

    ```json
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
                "args": ["./configs/bevdet/bevdet-r50.py"],
                // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"],
                "justMyCode": false

            }
        ]
    }
    ```

## 多卡训练

方法1. 命令行模式

```bash
bash tools/dist_train.sh ./configs/bevdet/bevdet-r50.py 3
```

方法2. vscode launch.json模式

```json
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
                "./configs/bevdet/bevdet-r50.py",
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

## 单GPU测试

测试时需要下载对应的pth文件
参考<https://github.com/WangYueFt/detr3d>中Evaluation部分，下载对应的pth文件，放置在ckpt文件夹下

方法1. 命令行模式

```bash
python tools/test.py ./configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/latest.pth --eval mAP
```

方法2. vscode launch.json模式

```json
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
                "configs/bevdet/bevdet-r50.py",
                "work_dirs/bevdet-r50/latest.pth",
                "--eval=mAP"
                ],
            "justMyCode": false
        }
    ]
}
```

## 多GPU测试

方法1. 命令行模式

```bash
bash ./tools/dist_test.sh ./configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/latest.pth 3 --eval mAP
```

方法2. vscode launch.json模式

```json
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
                "configs/bevdet/bevdet-r50.py",
                "work_dirs/bevdet-r50/latest.pth",
                "--eval=mAP"
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

## tensorboard

```bash
pip install protobuf==3.14.0
tensorboard --logdir work_dirs/bevdet-r50/tf_logs/
```

## 可视化

1. 生成json文件

```bash
# 运行test.py 必须--out", "--eval", "--format-only", "--show" or "--show-dir至少跟一个
# json文件生成需要增加 --eval-options参数 jsonfile_prefix=test_dirs
# 实在搞不清楚请看test.py的源码，看如何加载参数即可

# 1 直接测试
python tools/test.py ./configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/latest.pth --format-only

# 2 测试保存json文件
python tools/test.py ./configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/latest.pth --format-only --eval-options jsonfile_prefix=test_dirs
# 保留json位于目录test_dirs下

# 3 直接生成保存为pkl格式
python tools/test.py ./configs/bevdet/bevdet-r50.py ckpts/bevdet-r50.pth --out=./test_dirs/out.pkl
```

运行上面第2条指令生成./work_dir/results_nusc.json文件

2. json文件转可视化

```bash
python tools/analysis_tools/vis.py ./test_dirs/pts_bbox/results_nusc.json

```
