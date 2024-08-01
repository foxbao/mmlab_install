[TOC] 
# Create the conda environment
```
conda create --name mmdetection3d python=3.8 -y
conda activate mmdetection3d
```
# Something important to avoid bug 
https://github.com/open-mmlab/mmdetection/issues/10962
```
pip install numpy==1.23.5
pip install yapf==0.40.1
```

# Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
# Install mmcv

```
pip install -U openmim
mim install "mmcv>=2.0.0"
```
# Download mmdetection3d
```
git clone https://github.com/open-mmlab/mmdetection3d
```
# Install mmengine
Download and Install mmengine from source code inside the folder of mmdetection3d

```
cd mmdetection3d
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```
# Install mmdetection
Download and install mmdetection from source code inside the folder of mmdetection3d

Install mmdetection
```
cd mmdetection3d
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

# Install mmdetection3d
```
cd mmdetection3d
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