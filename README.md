# Exceed-T-Lab-S1
# 1.VAR
## 简介
在这里简要介绍你的项目，包括用途、功能和主要特点。

## 快速开始
VPN加速：
  ```bash
  source /etc/network_turbo
  ```
1. 克隆仓库：
   ```bash​
   git clone https://github.com/FoundationVision/VAR.git
   ```
2. 环境配置：
   ```bash
    conda create -n var python=3.10
    conda activate var
   ```
   Install ```torch>=2.0.0```
   Install other pip packages via ```pip3 install -r requirements.txt.```

## Imagenet数据准备
1. 创建空文件夹
```bash
mkdir train
mkdir val
```
2. 解压库里自带的数据集压缩包
```bash
tar xvf ../autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar -C ./train

tar xvf ../autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar -C ./val

tar -xzf ../autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz
```

3. 映射处理
```python
for x in ./train/*.tar; do mkdir -p "./train/$(basename "$x" .tar)" && tar -xvf "$x" -C "./train/$(basename "$x" .tar)" && rm "$x" && echo "已处理: $x"; done
```


```python
from scipy import io
import os
import shutil
 
def move_valimg(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    
    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))
 
        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))
 
if __name__ == '__main__':
    move_valimg()

```

## 训练
```bash
# d16, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
```
## 评估
### Sampling
For FID evaluation, use ```var.autoregressive_infer_cfg(..., cfg=1.5, top_p=0.96, top_k=900, more_smooth=False)``` to sample 50,000 images (50 per class) and save them as PNG (not JPEG) files in a folder. Pack them into a ```.npz``` file via ```create_npz_from_sample_folder(sample_folder)``` in utils/misc.py#L344. 
### Evaluate
Use the ```OpenAI's FID evaluation toolkit``` and reference ground truth npz file of ```256x256``` or ```512x512``` to evaluate ```FID, IS, precision, and recall```.


# 2.MAR
# 3.FAR
## 贡献

欢迎贡献代码！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 许可证

[MIT](LICENSE)
