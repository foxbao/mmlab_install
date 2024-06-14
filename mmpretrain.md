[TOC] 
# Create the conda environment
```
conda create --name mmpretrain python=3.8 -y
conda activate mmpretrain
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
# Download mmpretrain
```
git clone https://github.com/open-mmlab/mmpretrain.git
```
# Install mmengine
Download and Install mmengine from source code inside the folder of mmpretrain

```
cd mmpretrain
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```
# Install mmpretrain
```
cd mmpretrain
pip install -v -e .
```
