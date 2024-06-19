[TOC] 

https://blog.csdn.net/weixin_42545475/article/details/132422665
# Create the conda environment
```
conda create --name detr python=3.8 -y
conda activate detr
```




# Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


```
# Download detr3d
```
git clone https://github.com/facebookresearch/detr.git
```

# Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
