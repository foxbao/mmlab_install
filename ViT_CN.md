[TOC] 
参考了以下网页
https://blog.csdn.net/qq_45907168/article/details/138803978
https://blog.csdn.net/h904798869/article/details/132856083

# 安装 CUDA 11.3
要注意安装的CUDA版本要和之后安装的Pytorch版本一致
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

# 创建conda环境
```
conda create -n Vit python=3.8 -y
conda activate Vit
```



# 安装 Pytroch 1.10.0
切记，这里面的cudatoolkit=11.3，一定要和上面安装的CUDA版本一致
```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

```

# 下载mmpretrain源码
```
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -v e .
```

# 测试mmpretrain
```
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k
```


