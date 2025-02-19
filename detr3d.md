[TOC] 

https://blog.csdn.net/weixin_42545475/article/details/132422665

https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html

# 安装 gcc 12 and g++ 12
参考了以下网页
https://blog.csdn.net/qq_42059060/article/details/135677793
```
sudo apt-get install gcc-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
```

# Install CUDA 11.3
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

# Create the conda environment
```
conda create -n detr3d python=3.8 -y
conda activate detr3d
```

# Install Pytroch 1.9.0 based on cuda 11.3

```
切记，这里面的cudatoolkit=11.3，一定要和上面安装的CUDA版本一致
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html



import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

```

# Download detr3d
```
git clone https://github.com/WangYueFt/detr3d.git

```

# Something important to avoid bug 
https://github.com/open-mmlab/mmdetection/issues/10962
```
pip install numpy==1.23.5
pip install yapf==0.40.1
```

# Download mmdetection3d v1.0.0rc6
We can use the submodule of mmdetection3d in detr, with
```
git submodule init
git submodule update
```
Or download it from github
```
git clone https://github.com/open-mmlab/mmdetection3d.git
```

change tag
```
cd mmdetection3d
git tag
git checkout -b v1.0.0rc6 v1.0.0rc6
```

# Download and Install mmseg 0.30.0
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git tag
git checkout -b v0.30.0 v0.30.0
pip install -v -e .
```

or
```
pip install mmsegmentation==0.30.0


```

# Download and Install mmdetection 2.28.0
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git tag
git checkout -b v2.28.0 v2.28.0
pip install -v -e .
```

or
```
pip install -U openmim
mim install mmdet==2.28.0

```

# Download and Install mmengine 0.7.1
```
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
git tag
git checkout -b v0.7.1 v0.7.1
pip install -v -e .
```
or
```
mim install mmengine==0.7.1
```

# Install mmcv v1.6.0
https://github.com/open-mmlab/mmcv/issues/1386

```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
or
```
pip install -U openmim
mim install mmcv-full==1.6.0
```
# Install mmdetection3d v1.0.0rc6
```
git checkout v1.0.0rc6
cd mmdetection3d
pip install -v -e .
```


In case there is an error saying the mmcv version 1.6.0 is too high, we can change the following file to remove the error
mmdetection3d/mmdet3d/__init__.py
line 22
mmcv_maximum_version = '1.4.0'=>mmcv_maximum_version = '1.6.0'

# Verification of mmdection3d
```
conda install -c conda-forge libstdcxx-ng
pip install open3d
pip install -U openmim
mim download mmdet3d --config hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class --dest .
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth --show
```
If mmdetection3d works well, we will see a vehicle 3d detection result
If we see an error of KeyError: 'SparseConv2d is already registered in conv layer'
https://blog.csdn.net/qq_45779334/article/details/125145820
Change mmdet3d/ops/spconv/conv.py
Replace all 
@CONV_LAYERS.register_module()
to 
@CONV_LAYERS.register_module(force=True)

# process nuscene data
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
Notice: remember to uncomment line 224 in create_data.py, or the training data will not be generated


# Evaluation using pretrained backbone
Download the backbone file detr3d_resnet101.pth
```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py detr3d_resnet101.pth 1 --eval=bbox
```

if we have multiple gpus, we can use
```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py detr3d_resnet101.pth 3 --eval=bbox
```

if we want to save the result, we can use


if we want to visualize the result, we can use

```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py detr3d_resnet101.pth 3 --eval=bbox --out result.pkl
```

To use vscode for debugging
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
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": ["projects/configs/detr3d/detr3d_res101_gridmask.py","detr3d_resnet101.pth","--eval","box","--out","result.pkl"],
        }
    ]
}


# train on nuscenes data with one GPU
```
python tools/train.py 
```

```
"args": ["projects/configs/detr3d/detr3d_res101_gridmask.py","--cfg-options","load_from=pretrained/fcos3d.pth","--gpus","1"],

```

# Train on nuscenes data with multiple gpu
https://blog.csdn.net/XCCCCZ/article/details/134295931
We need to first change the nuscenes a little bit 
/home/ubuntu/anaconda3/envs/detr3d/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py
line 39
```
# self.class_names = self.class_range.keys()
self.class_names = list(self.class_range.keys())
```
Then Run the training code
```
tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 2
```


# Test on nuscenes data

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
