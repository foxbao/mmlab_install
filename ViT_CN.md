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


# 安装 mmcv v2.0.0

```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

```
# 安装 mmengine==0.8.3
```
pip install mmengine==0.8.3
```

# 下载mmpretrain源码
```
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -v e .
```

# 安装一些其他库
```
pip install setuptools==59.5.0
pip install yapf==0.40.1
```

# 测试mmpretrain
```
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k
```

# 使用ViT预测测试
建立一个test.py文件，贴入以下代码
```
from mmpretrain import inference_model
predict = inference_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

# 使用Vit
建立一个use_model.py,贴入以下代码
```
import torch
from mmpretrain import get_model

model = get_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

# vscode debug
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "use_model.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}
```

# 单GPU训练

1. 命令行模式
```
python tools/train.py configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py
```

2. vscode的launch.json配置模式
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "tools/train.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": ["configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py"],
            // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"],
            "justMyCode": false

        }
    ]
}

```



# 多卡训练

方法1. 命令行模式
Then Run the training code
```
bash tools/dist_train.sh configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py 3
```

方法2. vscode launch.json模式
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            // "program": "ddp_test.py",
            "console": "integratedTerminal",
            "module": "torch.distributed.run",
            "args": [
                "--nproc_per_node", "3",
                "tools/train.py",
                "--launcher=pytorch",
                "configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py",
                // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"
            ],
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}

```

# 单GPU测试
方法1. 命令行模式
```
python tools/test.py configs/vision_transformer/vit-base-p32_64xb64_in1k-384px.py https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth

```
方法2. vscode launch.json模式
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "tools/test.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "args": [
                "projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py",
                "ckpt/detr3d_resnet101.pth",
                "--eval=bbox"
                ],
            ],
            "justMyCode": false

        }
    ]
}

```

# 多GPU测试

方法1. 命令行模式
```
tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/ckpt 3 --eval=bbox
```

方法2. vscode launch.json模式
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            // "program": "ddp_test.py",
            "console": "integratedTerminal",
            "module": "torch.distributed.run",
            "args": [
                "--nproc_per_node", "3",
                "tools/test.py",
                "--launcher=pytorch",
                "projects/configs/detr3d/detr3d_res101_gridmask.py",
                "ckpt/detr3d_resnet101.pth",
                "--eval=box"
                // "--resume-from","./work_dirs/detr3d_res101_gridmask_cbgs/latest.pth"
            ],
            "env":{
                "PYTHONPATH":"${workspaceFolder}"
            },
            "justMyCode": false
        }
    ]
}

