[TOC] 
参考了以下网页
https://blog.csdn.net/qq_45907168/article/details/138803978
https://blog.csdn.net/h904798869/article/details/132856083

# 安装 CUDA 11.3
要注意安装的CUDA版本要和之后安装的Pytorch版本一致
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

# 创建conda环境
```
conda create -n Sparse4D python=3.8 -y
conda activate Sparse4D
```

# 下载Sparse4D源码
```
git clone https://github.com/HorizonRobotics/Sparse4D.git
```

# 安装 Pytroch 1.10.0
切记，这里面的cudatoolkit=11.3，一定要和上面安装的CUDA版本一致
```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

```

# 安装 mmcv
```
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

# 安装 mmdet==2.28.2


方法1. 通过pip安装(选择这种)
```
pip install mmdet==2.28.2
```

# 安装第三方库
```
pip install filelock
pip install urllib3==1.26.16
pip install pyquaternion==0.9.9
pip install nuscenes-devkit==1.1.10
pip install yapf==0.33.0
pip install tensorboard==2.14.0
pip install motmetrics==1.1.3
pip install pandas==1.1.5
pip install numpy==1.23.5
```

# 安装算子
```
cd projects/mmdet3d_plugin/ops
python setup.py develop
```


# 处理 nuscenes data
当我们用nuscenes数据进行训练时，需要先下载nuscenes数据集，然后进行预处理，生成训练用的数据集
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html
同时maptr要求can_bus数据,并且将map extension的数据放入nuscenes的maps文件夹中
```
cd Sparse4D
mkdir data
ln -s ~/Downloads/nuscenes ./data/nuscenes
pkl_path="data/nuscenes_anno_pkls"
mkdir -p ${pkl_path}
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

运行完之后nuscenes文件夹应该是如下结构
~/Downloads/test_detr3d/detr3d/data/nuscenes$ tree -L 1
MapTR
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl

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
pkl_path="data/nuscenes_anno_pkls"
mkdir -p ${pkl_path}
python3 tools/nuscenes_converter.py  --root_path ./data/nuscenes-mini --version v1.0-mini --info_prefix ${pkl_path}/nuscenes-mini
```
# 通过K-means生成anchors
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 tools/anchor_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
```

# 准备预训练模型
```
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

# 单GPU训练

1. 命令行模式
```
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704
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
            "args": ["projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"],
            // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"],
            "justMyCode": false

        }
    ]
}
```


# 多GPU训练
1. 命令行模式
修改local_train.sh，启动三个三卡0,1,2
```
export CUDA_VISIBLE_DEVICES=0,1,2
```

```
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704
```

1. vscode launch.json模式
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
                "projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py",
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
方法1. 命令行模式
修改local_test.sh代码
```
export CUDA_VISIBLE_DEVICES=0
```
```
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704  pretrained/sparse4dv3_r50.pth
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
                "projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py",
                "pretrained/sparse4dv3_r50.pth",
                "--eval=bbox"
                ],
            "justMyCode": false

        }
    ]
}
```



# 多GPU测试

方法1. 命令行模式
修改local_test.sh代码
```
export CUDA_VISIBLE_DEVICES=0,1,2
```
```
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704  pretrained/sparse4dv3_r50.pth
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
                "projects/configs/maptr/maptr_tiny_r50_24e.py",
                "pretrained/maptr_tiny_r50_24e.pth",
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
我们可以从代码网站下下载训练好的网络pth，跑对应的网络，来做可视化

方法1. 命令行模式
```
python tools/maptr/vis_pred.py projects/configs/maptr/maptr_tiny_r50_24e.py pretrained/maptr_tiny_r50_24e.pth

python tools/maptr/vis_pred.py projects/configs/maptr/maptr_tiny_r50_110e.py pretrained/maptr_tiny_r50_110e.pth

python tools/maptr/vis_pred.py projects/configs/maptr/maptr_tiny_fusion_24e.py pretrained/maptr_tiny_fusion_24e.pth
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
            "program": "tools/maptr/vis_pred.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": [
                "projects/configs/maptr/maptr_tiny_r50_24e.py",
                "pretrained/maptr_tiny_r50_24e.pth"
                ],
            "justMyCode": false

        }
    ]
}


```


