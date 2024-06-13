[TOC]


# Create the conda environment
```
conda create --name mmdetection3d python=3.8 -y
conda activate mmdetection3d
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
Download mmdetection3d
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```
# Install mmengine
Download and Install mmengine from source code inside the folder of mmdetection3d

```
cd mmdetection3d
https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```
# Install mmdetection
Download and install mmdetection from source code inside the folder of mmdetection3d

Install mmdetection
```
cd mmdetection
pip install -v -e .
```