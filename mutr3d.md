[TOC] 
# Create the conda environment
```
conda create --name detr3d python=3.8 -y
conda activate mutr3d
```

# Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
# Download mutr3d
```
git clone https://github.com/WangYueFt/detr3d.git
```


# Install mmcv

```
pip install -U openmim
mim install "mmcv==1.3.14"
```

# Install mmdetection 2.12.0
Download from github 
https://github.com/open-mmlab/mmdetection/releases/tag/v2.12.0
```
cd mmdetection-2.12.0
pip install -v -e .
```

# nuscenes-devkit
```
pip install nuscenes-devkit
```
# 
```
pip install motmetrics==1.1.3
```


# Install mmengine
Download and Install mmengine from source code inside the folder of detr3d

```
cd detr3d
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```
# Install mmdetection
Download and install mmdetection from source code inside the folder of mmdetection3d

Install mmdetection
```
cd detr3d
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

# Download and install mmdetection3d
```
git clone https://github.com/open-mmlab/mmdetection3d
cd mmdetection3d
pip install -e . -v
```


In case there is an error saying the mmcv version is too high, we can change the following file to remove the error
mmdetection/mmdet/__init__.py
line 16
```
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version))
```
