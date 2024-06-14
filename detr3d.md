[TOC] 

https://blog.csdn.net/weixin_42545475/article/details/132422665
# Create the conda environment
```
conda create --name detr3d python=3.8 -y
conda activate detr3d
```


pip install -U openmim

mim install mmdet==2.28.1

mim install mmsegmentation==0.28.0

mim install mmcv-full==1.6.0




# Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116


```
# Download detr3d
```
git clone https://github.com/WangYueFt/detr3d.git
```


# Install mmcv

```
pip install -U openmim
mim install "mmcv>=2.0.0"
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
