# 前言

本章我们来介绍如何使用PaddlePaddle训练一个区分不同音频的分类模型，例如你有这样一个需求，需要根据不同的鸟叫声识别是什么种类的鸟，这时你就可以使用这个方法来实现你的需求了。


**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`758170167`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


# 使用准备

 - Anaconda 3
 - Python 3.8
 - PaddlePaddle 2.4.0
 - Windows 10 or Ubuntu 18.04

# 项目特性

1. 支持模型：EcapaTdnn、PANNS、TDNN、Res2Net、ResNetSE
2. 支持池化层：AttentiveStatisticsPooling(ASP)、SelfAttentivePooling(SAP)、TemporalStatisticsPooling(TSP)
   、TemporalAveragePooling(TAP)
3. 支持预处理方法：MelSpectrogram、LogMelSpectrogram、Spectrogram、MFCC、Fbank


# 模型测试表

|      模型      | Params(M) | 预处理方法 |     数据集      | 类别数量 |   准确率   |
|:------------:|:---------:|:-----:|:------------:|:----:|:-------:|
|  EcapaTdnn   |    6.2    | Flank | UrbanSound8K |  10  | 0.90503 |
| PANNS（CNN10） |    4.9    | Flank | UrbanSound8K |  10  | 0.95454 |
|     TDNN     |    2.7    | Flank | UrbanSound8K |  10  | 0.92045 |
|   Res2Net    |    5.6    | Flank | UrbanSound8K |  10  | 0.85812 |
|   ResNetSE   |    9.1    | Flank | UrbanSound8K |  10  | 0.92219 |
|   ERes2Net   |    9.1    | Flank | UrbanSound8K |  10  |         |
|   CAMPPlus   |    9.1    | Flank | UrbanSound8K |  10  | 0.96590 |


## 安装环境

 - 首先安装的是PaddlePaddle的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.4.0 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

 - 安装ppacls库。
 
使用pip安装，命令如下：
```shell
python -m pip install ppacls -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/AudioClassification_PaddlePaddle.git
cd AudioClassification_PaddlePaddle
python setup.py install
```

## 数据数据

生成数据列表，用于下一步的读取需要，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒以上，如 `dataset/audio/鸟叫声/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

Urbansound8K 是目前应用较为广泛的用于自动城市环境声分类研究的公共数据集，包含10个分类：空调声、汽车鸣笛声、儿童玩耍声、狗叫声、钻孔声、引擎空转声、枪声、手提钻、警笛声和街道音乐声。数据集下载地址：[UrbanSound8K.tar.gz](https://aistudio.baidu.com/aistudio/datasetdetail/36625)。以下是针对Urbansound8K生成数据列表的函数。如果读者想使用该数据集，请下载并解压到 `dataset`目录下，把生成数据列表代码改为以下代码。

执行`create_data.py`即可生成数据列表，里面提供了两种生成列表方式，第一种是自定义的数据，第二种是生成Urbansound8K的数据列表，具体看代码。
```shell
python create_data.py
```

生成的列表是长这样的，前面是音频的路径，后面是该音频对应的标签，从0开始，路径和标签之间用Tab隔开。
```shell
dataset/UrbanSound8K/audio/fold2/104817-4-0-2.wav	4
dataset/UrbanSound8K/audio/fold9/105029-7-2-5.wav	7
dataset/UrbanSound8K/audio/fold3/107228-5-0-0.wav	5
dataset/UrbanSound8K/audio/fold4/109711-3-2-4.wav	3
```

# 修改预处理方法

配置文件中默认使用的是MelSpectrogram预处理方法，如果要使用其他预处理方法，可以修改配置文件中的安装下面方式修改，具体的值可以根据自己情况修改。如果不清楚如何设置参数，可以直接删除该部分，直接使用默认值。

```yaml
preprocess_conf:
  # 音频预处理方法，支持：MelSpectrogram、Spectrogram、MFCC、Fbank
  feature_method: 'MelSpectrogram'
  # 设置API参数，更参数查看对应API，不清楚的可以直接删除该部分，直接使用默认值
  method_args:
    sample_rate: 16000
    n_fft: 1024
    hop_length: 320
    win_length: 1024
    f_min: 50.0
    f_max: 14000.0
    n_mels: 64
```

## 训练

接着就可以开始训练模型了，创建 `train.py`。配置文件里面的参数一般不需要修改，但是这几个是需要根据自己实际的数据集进行调整的，首先最重要的就是分类大小`dataset_conf.num_class`，这个每个数据集的分类大小可能不一样，根据自己的实际情况设定。然后是`dataset_conf.batch_size`，如果是显存不够的话，可以减小这个参数。

```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
python -m paddle.distributed.launch --gpus '0,1' train.py
```
训练输出日志：
```
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:14 - ----------- 额外配置参数 -----------
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:16 - configs: configs/ecapa_tdnn.yml
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:16 - pretrained_model: None
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:16 - resume_model: None
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:16 - save_model_path: models/
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:16 - use_gpu: True
[2023-08-07 23:02:08.807036 INFO   ] utils:print_arguments:17 - ------------------------------------------------
[2023-08-07 23:02:08.811036 INFO   ] utils:print_arguments:19 - ----------- 配置文件参数 -----------
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:22 - dataset_conf:
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:25 - 	aug_conf:
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		noise_aug_prob: 0.2
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		noise_dir: dataset/noise
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		speed_perturb: True
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		volume_aug_prob: 0.2
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		volume_perturb: False
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:25 - 	dataLoader:
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		batch_size: 64
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		num_workers: 4
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	do_vad: False
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:25 - 	eval_conf:
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		batch_size: 1
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		max_duration: 20
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	label_list_path: dataset/label_list.txt
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	max_duration: 3
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	min_duration: 0.5
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	sample_rate: 16000
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:25 - 	spec_aug_args:
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		freq_mask_width: [0, 8]
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:27 - 		time_mask_width: [0, 10]
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	target_dB: -20
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	test_list: dataset/test_list.txt
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	train_list: dataset/train_list.txt
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	use_dB_normalization: True
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:29 - 	use_spec_aug: True
[2023-08-07 23:02:08.812035 INFO   ] utils:print_arguments:22 - model_conf:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	num_class: 10
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	pooling_type: ASP
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:22 - optimizer_conf:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	optimizer: Adam
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	scheduler: WarmupCosineSchedulerLR
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:25 - 	scheduler_args:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:27 - 		learning_rate: 0.001
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:27 - 		min_lr: 1e-05
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:27 - 		warmup_epoch: 5
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	weight_decay: 1e-06
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:22 - preprocess_conf:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	feature_method: Fbank
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:25 - 	method_args:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:27 - 		n_mels: 80
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:27 - 		sr: 16000
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:22 - train_conf:
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	log_interval: 10
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:29 - 	max_epoch: 60
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:31 - use_model: EcapaTdnn
[2023-08-07 23:02:08.816062 INFO   ] utils:print_arguments:32 - ------------------------------------------------
[2023-08-07 23:02:08.817077 WARNING] trainer:__init__:69 - Windows系统不支持多线程读取数据，已自动关闭！
W0807 23:02:08.822477  3192 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.7, Runtime API Version: 11.6
W0807 23:02:08.826478  3192 gpu_resources.cc:91] device: 0, cuDNN Version: 8.4.
----------------------------------------------------------------------------------------
        Layer (type)             Input Shape          Output Shape         Param #    
========================================================================================
          Conv1D-2              [[1, 80, 102]]        [1, 512, 98]         205,312    
          Conv1d-1              [[1, 80, 98]]         [1, 512, 98]            0       
           ReLU-1               [[1, 512, 98]]        [1, 512, 98]            0       
       BatchNorm1D-2            [[1, 512, 98]]        [1, 512, 98]          2,048     
       BatchNorm1d-1            [[1, 512, 98]]        [1, 512, 98]            0       
        TDNNBlock-1             [[1, 80, 98]]         [1, 512, 98]            0       
          Conv1D-4              [[1, 512, 98]]        [1, 512, 98]         262,656    
          Conv1d-3              [[1, 512, 98]]        [1, 512, 98]            0       
           ReLU-2               [[1, 512, 98]]        [1, 512, 98]            0       
       BatchNorm1D-4            [[1, 512, 98]]        [1, 512, 98]          2,048     
       BatchNorm1d-3            [[1, 512, 98]]        [1, 512, 98]            0       
        TDNNBlock-2             [[1, 512, 98]]        [1, 512, 98]            0       
          Conv1D-6              [[1, 64, 102]]        [1, 64, 98]          12,352     
          Conv1d-5              [[1, 64, 98]]         [1, 64, 98]             0       
           ReLU-3               [[1, 64, 98]]         [1, 64, 98]             0       
       BatchNorm1D-6            [[1, 64, 98]]         [1, 64, 98]            256      
       BatchNorm1d-5            [[1, 64, 98]]         [1, 64, 98]             0       
        TDNNBlock-3             [[1, 64, 98]]         [1, 64, 98]             0       
          Conv1D-8              [[1, 64, 102]]        [1, 64, 98]          12,352     
          Conv1d-7              [[1, 64, 98]]         [1, 64, 98]             0       
           ReLU-4               [[1, 64, 98]]         [1, 64, 98]             0       
       BatchNorm1D-8            [[1, 64, 98]]         [1, 64, 98]            256      
       BatchNorm1d-7            [[1, 64, 98]]         [1, 64, 98]             0       
        TDNNBlock-4             [[1, 64, 98]]         [1, 64, 98]             0       
         Conv1D-10              [[1, 64, 102]]        [1, 64, 98]          12,352     
          Conv1d-9              [[1, 64, 98]]         [1, 64, 98]             0       
           ReLU-5               [[1, 64, 98]]         [1, 64, 98]             0       
       BatchNorm1D-10           [[1, 64, 98]]         [1, 64, 98]            256      
       BatchNorm1d-9            [[1, 64, 98]]         [1, 64, 98]             0       
        TDNNBlock-5             [[1, 64, 98]]         [1, 64, 98]             0       
         Conv1D-12              [[1, 64, 102]]        [1, 64, 98]          12,352     
         Conv1d-11              [[1, 64, 98]]         [1, 64, 98]             0       
           ReLU-6               [[1, 64, 98]]         [1, 64, 98]             0       
       BatchNorm1D-12           [[1, 64, 98]]         [1, 64, 98]            256      
       BatchNorm1d-11           [[1, 64, 98]]         [1, 64, 98]             0       
        TDNNBlock-6             [[1, 64, 98]]         [1, 64, 98]             0      
······················································  
       BatchNorm1d-59           [[1, 128, 98]]        [1, 128, 98]            0       
        TDNNBlock-30           [[1, 4608, 98]]        [1, 128, 98]            0       
           Tanh-1               [[1, 128, 98]]        [1, 128, 98]            0       
         Conv1D-74              [[1, 128, 98]]       [1, 1536, 98]         198,144    
         Conv1d-73              [[1, 128, 98]]       [1, 1536, 98]            0       
AttentiveStatisticsPooling-1   [[1, 1536, 98]]        [1, 3072, 1]            0       
       BatchNorm1D-62           [[1, 3072, 1]]        [1, 3072, 1]         12,288     
       BatchNorm1d-61           [[1, 3072, 1]]        [1, 3072, 1]            0       
         Conv1D-76              [[1, 3072, 1]]        [1, 192, 1]          590,016    
         Conv1d-75              [[1, 3072, 1]]        [1, 192, 1]             0       
          Linear-1                [[1, 192]]            [1, 10]             1,930     
========================================================================================
Total params: 6,215,306
Trainable params: 6,195,978
Non-trainable params: 19,328
----------------------------------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 35.53
Params size (MB): 23.71
Estimated Total Size (MB): 59.27
----------------------------------------------------------------------------------------

[2023-08-07 23:02:11.081835 INFO   ] trainer:train:317 - 训练数据：8644
[2023-08-07 23:02:15.428326 INFO   ] trainer:__train_epoch:269 - Train epoch: [1/60], batch: [0/136], loss: 2.99582, accuracy: 0.04688, learning rate: 0.00000000, speed: 14.72 data/sec, eta: 9:51:07
```


# 评估
每轮训练结束可以执行评估，评估会出来输出准确率，还保存了混合矩阵图片，保存路径`output/images/`，如下。
![混合矩阵](docs/images/image1.png)

# 预测

在训练结束之后，我们得到了一个模型参数文件，我们使用这个模型预测音频。

```shell
python infer.py --audio_path=dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav
```

# 其他功能

 - 为了方便读取录制数据和制作数据集，这里提供了录音程序`record_audio.py`，这个用于录制音频，录制的音频采样率为16000，单通道，16bit。

```shell
python record_audio.py
```

 - `infer_record.py`这个程序是用来不断进行录音识别，我们可以大致理解为这个程序在实时录音识别。通过这个应该我们可以做一些比较有趣的事情，比如把麦克风放在小鸟经常来的地方，通过实时录音识别，一旦识别到有鸟叫的声音，如果你的数据集足够强大，有每种鸟叫的声音数据集，这样你还能准确识别是那种鸟叫。如果识别到目标鸟类，就启动程序，例如拍照等等。

```shell
python infer_record.py --record_seconds=3
```

## 打赏作者

<br/>
<div align="center">
<p>打赏一块钱支持一下作者</p>
<img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>
