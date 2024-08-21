# 运行方法和参数设置
## 1. 程序安装与获取
直接使用git clone即可方便地获取本程序：
``` bash
git clone https://github.com/Tyrion58/SZD_image_generate
```
## 2. 运行环境配置
使用的包种类较多，用户可以直接使用我自己导出的Anaconda运行环境`env.yml`，只需要在本程序的根目录下运行：

```bash
conda env create -n SZD -f env.yml
```
即可获取我的运行环境。

然后激活运行环境：
```bash
conda activate SZD
```
## 3.开始训练
首先请在https://wandb.ai 注册一个你自己的wandb平台账号，获取自己的账号识别码，然后运行：
```bash
wandb login
```

根据命令行提示输入自己的识别码即可完成登录。然后运行
```bash
python train.py
```
即可开始训练，如果需要对参数进行自定义设置，请使用帮助指令：
```bash
python train.py -h
```
你会看到每一个可选参数的具体解释，然后根据需要决定是否更改默认值。
```bash
D:\SZD_cgan\SZD_image_generate>python train.py -h
usage: train.py [-h] [--model_name MODEL_NAME] [--data_name DATA_NAME] [--data_path DATA_PATH]
                [--image_subpath IMAGE_SUBPATH] [--checkpoint_dir CHECKPOINT_DIR] [--epochs EPOCHS] [-b BATCH_SIZE]
                [--lr LR] [--beta1 BETA1] [--image_size IMAGE_SIZE] [--channels CHANNELS] [--netD NETD] [--netG NETG]
                [--train TRAIN]

Train

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model architecture
  --data_name DATA_NAME
                        the name of the data
  --data_path DATA_PATH
                        Path to dataset.
  --image_subpath IMAGE_SUBPATH
                        Path to dataset.
  --checkpoint_dir CHECKPOINT_DIR
                        the directory to save the model
  --epochs EPOCHS       Number of total epochs to run. (Default: 30)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size of the dataset. (Default: 32)
  --lr LR               Learning rate. (Default: 0.0002)
  --beta1 BETA1         beta1 of Adam. (Default: 0.5)
  --image_size IMAGE_SIZE
                        Image size. (Default: 64)
  --channels CHANNELS   The number of channels of the image. (Default: 1)
  --netD NETD           Path to Discriminator checkpoint.
  --netG NETG           Path to Generator checkpoint.
  --train TRAIN         whether to load the pretrained model
```
训练过程中可以打开命令行中出现的网址对训练过程进行实时监测。

## 4. 模型效果评估
在根目录下运行
```bash
python eval --model_dir YOURMODEL
```
即可对模型进行测试，与`train.py`一样，你也可以使用`python eval -h`获取帮助。

运行结束后会在根目录下生成一个`eval_image`文件夹，本次测试的结果将会 以当前时间命名保存
