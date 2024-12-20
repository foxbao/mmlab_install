[TOC] 
参考了以下网页
https://blog.csdn.net/weixin_42545475/article/details/132422665

https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html

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

# 安装 Pytroch 1.11.0
切记，这里面的cudatoolkit=11.3，一定要和上面安装的CUDA版本一致
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

```

# 下载 detr3d
```
git clone https://github.com/WangYueFt/detr3d.git

```

# 安装numpy和yapf来避免一些bug
参考以下网页，需要提前安装numpy和yapf的特定版本，否则回报错
https://github.com/open-mmlab/mmdetection/issues/10962
```
pip install numpy==1.23.5
pip install yapf==0.40.1
```

# 下载 mmdetection3d v1.0.0rc6
detr3d代码中包含了mmdetection3d的子仓库，我们可以直接通过以下命令，拉取子模块的代码，注意要切换到v1.0.0rc6版本。安装mmdetection3d需要在后面一些库安装完之后再装
We can use the submodule of mmdetection3d in detr, with
```
git submodule init
git submodule update
git checkout -b v1.0.0rc6 v1.0.0rc6
```
或者
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git tag
git checkout -b v1.0.0rc6 v1.0.0rc6
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
方法2. 通过pip 安装
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

# 安装 mmcv v1.6.0

```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
```

# 安装 mmdetection3d v1.0.0rc6
前置所有依赖库安装完毕后，安装mmdetection3d
```
cd mmdetection3d
git checkout v1.0.0rc6
pip install -v -e .
```

如果报错说mmcv版本过高，我们可以修改mmdetection3d的__init__.py文件，将mmcv的版本检查去掉
mmdet/__init__.py
line 16
```
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version))
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

remember to uncomment line 224 in create_data.py

# 下载预训练的backbone


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
