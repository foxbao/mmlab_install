# MMdetection3d 1.1.0 配置指南

[TOC]

参考了以下网页
<https://blog.csdn.net/weixin_42545475/article/details/132422665>
<https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html>

## 安装 gcc 12 and g++ 12

参考了以下网页
<https://blog.csdn.net/qq_42059060/article/details/135677793>

```bash
sudo apt-get install gcc-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
```

## 安装nvidia驱动

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

## 安装 CUDA 11.3

要注意安装的CUDA版本要和之后安装的Pytorch版本一致

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3ash.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

如果出现gcc相关error，命令改为

```bash
sudo sh cuda_11.3.0_465.19.01_linux.run --override
```

## 创建conda环境

```bash
conda create -n mmdetection3d python=3.8 -y
conda activate mmdetection3d
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

## 安装mmengine

```bash
pip install mmengine
```

## 安装mmcv

```bash
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

## 安装mmdet

```bash
pip install mmdet==3.0.0
```

## 下载 mmdetection3d

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .
```

## 验证安装

下载配置文件和模型权重文件

```bash
pip install -U openmim
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
```

验证pointpillar模型

```bash
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```

## 推理验证

推理部分是采用api，来进行单帧数据的推理

1. 单模态激光

    ```python
    from mmdet3d.apis import init_model, inference_detector

    config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    model = init_model(config_file, checkpoint_file)
    inference_detector(model, 'demo/data/kitti/000008.bin')
    ```

2. 多模态样例

在 NuScenes 数据上测试 BEVFusion 模型

```bash
python projects/BEVFusion/setup.py develop
export PYTHONPATH=$PYTHONPATH:$(pwd)

python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py bevfusion_converted.pth --cam-type all --score-thr 0.2 --show
```

vscode launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "projects/BEVFusion/demo/multi_modality_demo.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": ["demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin","demo/data/nuscenes/","demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl","projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py","bevfusion_converted.pth","--cam-type=all","--score-thr=0.2","--show"],
            "justMyCode": false

        }
    ]
}
```
