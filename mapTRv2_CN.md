[TOC] 
参考了以下网页
https://blog.csdn.net/qq_45907168/article/details/138803978
https://blog.csdn.net/h904798869/article/details/132856083

# 安装 gcc 12 and g++ 12
参考了以下网页
https://blog.csdn.net/qq_42059060/article/details/135677793
```
sudo apt-get install gcc-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
```

# 安装 CUDA 11.3
要注意安装的CUDA版本要和之后安装的Pytorch版本一致
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

# 创建conda环境
```
conda create -n MapTRv2 python=3.8 -y
conda activate MapTRv2
```

# 安装 Pytroch 1.10.0
切记，这里面的cudatoolkit=11.3，一定要和上面安装的CUDA版本一致
```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html


```

# 下载 MapTRv2
```
git clone https://github.com/hustvl/MapTR.git MapTRv2
cd MapTRv2
git checkout maptrv2

```

# 安装numpy和yapf来避免一些bug
参考以下网页，需要提前安装numpy和yapf的特定版本，否则回报错
https://github.com/open-mmlab/mmdetection/issues/10962
```
pip install numpy==1.21.0
pip install yapf==0.40.1
```

# 安装 mmcv v1.4.0

```
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

```

# 下载安装 mmdetection 2.14.0


方法1. 通过pip安装(选择这种)
```
pip install mmdet==2.14.0
```

# 下载安装 mmsegmentation==0.14.1


方法2. 通过pip 安装
```
pip install mmsegmentation==0.14.1
```

# 安装 timm

```
pip install timm
```

# 安装mmdet3d和GKT
```
cd mmdetection3d
pip install -v e .
cd ..
cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```
# 安装其他功能包

```
cd MapTR
pip install -r requirement.txt 
```

# 降低setuptools和numpy版本
之前的安装可能会把setuptools的版本提升到60.2.0，甚至75.0，版本过高，会在后面运行detr3d时报错，因此需要降级到59.5.0
首先通过
```
pip list | grep setuptools
pip list | grep numpy
```
确定当前setuptools版本

```
pip install setuptools==59.5.0
pip install numpy==1.21.0
```



# 处理 nuscenes data
当我们用nuscenes数据进行训练时，需要先下载nuscenes数据集，然后进行预处理，生成训练用的数据集
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html
同时maptr要求can_bus数据,并且将map extension的数据放入nuscenes的maps文件夹中
```
cd detr3d
mkdir data
cd data
ln -s /path/to/nuscenes ./
ln -s /path/to/can_bus ./
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data

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
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes-mini --out-dir ./data/nuscenes-mini --extra-tag nuscenes --version v1.0-mini --canbus ./data
python tools/create_data.py nuscenes --root-path ./data/nuscenes-mini --out-dir ./data/nuscenes-mini --extra-tag nuscenes --version v1.0-mini

```

# 准备预训练模型backbone
```
cd /path/to/MapTR
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

# 单GPU训练

1. 命令行模式
```
bash ./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py 1

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
            "args": ["projects/configs/maptr/maptr_tiny_r50_24e.py"],
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

1. 命令行模式
Then Run the training code
```
bash ./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py 3

```

2. vscode launch.json模式
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
                "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py",
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
```
bash tools/dist_test_map.sh projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py pretrained/maptrv2_nusc_r50_24e.pth 1 --eval=bbox
```

方法2. vscode launch.json模式
!!!注意需要修改tools/test.py中代码
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
                "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py",
                "pretrained/maptrv2_nusc_r50_24e.pth",
                "--eval=bbox"
                ],
            "justMyCode": false

        }
    ]
}
```



# 多GPU测试

方法1. 命令行模式
```

bash tools/dist_test_map.sh projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py pretrained/maptrv2_nusc_r50_24e.pth 1 --eval=bbox
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
                "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py",
                "pretrained/maptrv2_nusc_r50_24e.pth",
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
我们可以从代码网站下下载训练好的网络pth放入pretrained，或者自己训练好的pth，跑对应的网络，来做可视化

方法1. 命令行模式
```
python tools/maptrv2/nusc_vis_pred.py projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py pretrained/maptrv2_nusc_r50_24e.pth

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
            "program": "tools/maptrv2/nusc_vis_pred.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": [
                "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py",
                "pretrained/maptrv2_nusc_r50_24e.pth"
                ],
            "justMyCode": false

        }
    ]
}


```


