[TOC] 

https://mmcv.readthedocs.io/en/v1.5.0/get_started/installation.html

# Install gcc 8 and g++ 8
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

# Install CUDA 11.3
```


wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run

```



# Create the conda environment
```
conda create -n mmdetection_v1.0.0rc6 python=3.8 -y
conda activate mmdetection_v1.0.0rc6
```

# Install Pytroch 11.1
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

```

# Install dependencies
```
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
# Install mmcv

```

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git tag
git checkout -b v1.7.0 v1.7.0

pip install -v -e .
```

# Install mmdetection3d
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git tag
git checkout -b v1.1.0 v1.1.0
pip install -v -e .
```

# Install mmdetection
```
        
`
pip install opencv-python
pip install openmim
mim install "mmcv==1.7.0"
```

# Download mmdetection3d v1.1.0
```
git clone https://github.com/open-mmlab/mmdetection3d
git tag
git checkout -b v1.1.0 v1.1.0
```
# Install mmengine
Download and Install mmengine from source code inside the folder of mmdetection3d

```
mim install mmengine

cd mmdetection3d
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```
# Install mmdetection v2.28.0
Download and install mmdetection from source code inside the folder of mmdetection3d
```
cd mmdetection3d
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git tag
git checkout -b v2.28.0 v2.28.0
pip install -v -e .
```

# Install mmseg
```
pip install "mmsegmentation>==0.30.0"

```

# Install mmdetection3d v1.0.0rc6
```
cd mmdetection3d
git tag 
git checkout -b v1.0.0rc6 v1.0.0rc6
pip install -v -e .
```


In case there is an error saying the mmcv version is too high, we can change the following file to remove the error
mmdet/__init__.py
line 16
```
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version))
```

# Verification
```
conda install -c conda-forge libstdcxx-ng
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
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