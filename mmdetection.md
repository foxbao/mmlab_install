
Create the conda environment
```
conda create --name mmdetection python=3.8 -y
conda activate mmdetection
```

Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Install mmcv

```
pip install -U openmim
mim install "mmcv>=2.0.0"
```

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```
Install mmengine from source code

```
cd mmdetection
https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```

Install mmdetection
```
cd mmdetection
pip install -v -e .
```

Change the file
mmdet/__init__.py
line 16
```
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version))
```
