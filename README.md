# 前言

本章我们来介绍如何使用PaddelPaddle训练一个区分不同音频的分类模型，例如你有这样一个需求，需要根据不同的鸟叫声识别是什么种类的鸟，这时你就可以使用这个方法来实现你的需求了。

# 环境准备

主要介绍libsora，PyAudio，pydub的安装，其他的依赖包根据需要自行安装。

- Python 3.7
- Tensorflow 2.0

## 安装libsora

最简单的方式就是使用pip命令安装，如下：

```shell
pip install pytest-runner
pip install librosa
```

如果pip命令安装不成功，那就使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/)， windows的可以下载zip压缩包，方便解压。

```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现 `libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如 `pip install librosa==0.6.3`

安装ffmpeg， 下载地址：[http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)，笔者下载的是64位，static版。
然后到C盘，笔者解压，修改文件名为 `ffmpeg`，存放在 `C:\Program Files\`目录下，并添加环境变量 `C:\Program Files\ffmpeg\bin`

最后修改源码，路径为 `C:\Python3.7\Lib\site-packages\audioread\ffdec.py`，修改32行代码，如下：

```python
COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
```

## 安装PyAudio

使用pip安装命令，如下：

```shell
pip install pyaudio
```

在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

## 安装pydub

使用pip命令安装，如下：

```shell
pip install pydub
```

# 训练分类模型

把音频转换成训练数据最重要的是使用了librosa，使用librosa可以很方便得到音频的梅尔频谱（Mel Spectrogram），使用的API为 `librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用tensorflow训练和预测。关于梅尔频谱具体信息读者可以自行了解，跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为 `librosa.feature.mfcc()`。同样以下的代码，就可以获取到音频的梅尔频谱，其中 `duration`参数指定的是截取音频的长度。

```python
y1, sr1 = librosa.load(data_path, duration=2.97)
ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
```

## 创建训练数据

我们训练的数据就是通过librosa把音频生成梅尔频谱的数据，但是生成梅尔频谱的数据时间比较长，如果过是边训练边生成，这样会严重影响训练的速度，所以最后是在训练前，我们把所有的训练数据都转换成梅尔频谱并存储在二进制文件中，这样不仅省去了生成梅尔频谱的时间，还能缩短读取文件的时间。当文件的数量非常多时，文件的读取就会变得非常慢，如果我们把这些文件写入到一个二进制文件中，这样读取速度将会大大提高。下面我们就来把音频数据生成我们所需的训练数据

首先需要生成数据列表，用于下一步的读取需要，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在 `dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在5秒左右，如 `dataset/audio/鸟叫声/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

```python
# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    persons = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    for i in range(len(persons)):
        sounds = os.listdir(os.path.join(audio_path, persons[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, persons[i], sound)
            t = librosa.get_duration(filename=sound_path)
            # 过滤小于3秒的音频
            if t >= 3:
                if sound_sum % 100 == 0:
                    f_test.write('%s\t%d\n' % (sound_path, i))
                else:
                    f_train.write('%s\t%d\n' % (sound_path, i))
                sound_sum += 1
        print("Person：%d/%d" % (i + 1, len(persons)))

    f_test.close()
    f_train.close()
   
if __name__ == '__main__':
    get_data_list('dataset/audio', 'dataset')
```

生成数据列表之后，下一步开始把这些音频生成梅尔频谱的二进制文件。生成的二进制文件有三个，`.data`是存放梅尔频谱数据的，全部的数据都存放在这个文件中，`.header`存放每条数据的key，`.label`存放数据的标签值，通过这个key之后可以获取 `.data`中的数据和 `.label`的标签，以及 `.data`中每条数据的偏移量。

```python
import os
import struct
import uuid
import librosa
import pandas as pd
from tqdm import tqdm


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_data(self, key, data):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(data)))
        self.data_file.write(data)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(data)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(data)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


# 格式二进制转换
def convert_data(data_list_path, output_prefix):
    # 读取列表
    data_list = open(data_list_path, "r").readlines()
    print("train_data size:", len(data_list))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for record in tqdm(data_list):
        try:
            path, label = record.replace('\n', '').split('\t')
            y1, sr1 = librosa.load(path, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y1, sr=sr1).reshape(-1).tolist()
            if len(ps) != 128 * 128: continue
            data = struct.pack('%sd' % len(ps), *ps)
            # 写入对应的数据
            key = str(uuid.uuid1())
            writer.add_data(key, data)
            writer.add_label('\t'.join([key, label.replace('\n', '')]))
        except Exception as e:
            print(e)
        
if __name__ == '__main__':
    convert_data('dataset/train_list.txt', 'dataset/train')
    convert_data('dataset/test_list.txt', 'dataset/test')
```

Urbansound8K 是目前应用较为广泛的用于自动城市环境声分类研究的公共数据集，包含10个分类：空调声、汽车鸣笛声、儿童玩耍声、狗叫声、钻孔声、引擎空转声、枪声、手提钻、警笛声和街道音乐声。数据集下载地址：[https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)。以下是针对Urbansound8K生成数据列表的函数。如果读者想使用该数据集，请下载并解压到 `dataset`目录下，把生成数据列表代码改为以下代码。

```python
# 创建UrbanSound8K数据列表
def get_urbansound8k_list(path, urbansound8k_cvs_path):
    data_list = []
    data = pd.read_csv(urbansound8k_cvs_path)
    # 过滤掉长度少于3秒的音频
    valid_data = data[['slice_file_name', 'fold', 'classID', 'class']][data['end'] - data['start'] >= 3]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    for row in valid_data.itertuples():
        data_list.append([row.path, row.classID])

    f_train = open(os.path.join(path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(path, 'test_list.txt'), 'w')

    for i, data in enumerate(data_list):
        sound_path = os.path.join('dataset/UrbanSound8K/audio/', data[0])
        if i % 100 == 0:
            f_test.write('%s\t%d\n' % (sound_path, data[1]))
        else:
            f_train.write('%s\t%d\n' % (sound_path, data[1]))

    f_test.close()
    f_train.close()


if __name__ == '__main__':
    get_urbansound8k_list('dataset', 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
```

创建 `reader.py`用于在训练时读取数据。编写一个 `ReadData`类，用读取上一步生成的二进制文件，通过 `.header`中的key和每条数据的偏移量，将 `.data`的数据读取出来，并通过key来绑定data和label的对应关系。

```python
import struct
import mmap
import numpy as np


class ReadData(object):
    def __init__(self, prefix_path):
        self.offset_dict = {}
        for line in open(prefix_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(prefix_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('loading label')
        # 获取label
        self.label = {}
        for line in open(prefix_path + '.label', 'rb'):
            key, label = line.split(b'\t')
            self.label[key] = [int(label.decode().replace('\n', ''))]
        print('finish loading data:', len(self.label))

    # 获取图像数据
    def get_data(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()
```

使用上面的工具，创建 `train_reader`和 `test_reader`函数，用于在训练读取训练数据和测试数据，`train_reader`多了 `np.random.shuffle(keys)`操作，作用是为了每一轮的训练，数据都是打乱的，使得每次一轮的输入数据顺序都不一样。

```python
def mapper(sample):
    data, label = sample
    data = list(struct.unpack('%sd' % (128 * 128), data))
    data = np.array(data).reshape((1, 128, 128)).astype(np.float32)
    assert (data is not None), 'data is None'
    return data, label


def train_reader(data_path, batch_size):
    def reader():
        readData = ReadData(data_path)
        keys = readData.get_keys()
        keys = list(keys)
        np.random.shuffle(keys)

        batch_data, batch_label = [], []
        for key in keys:
            data = readData.get_data(key)
            assert (data is not None)
            label = readData.get_label(key)
            assert (label is not None)
            sample = (data, label)
            d, label = mapper(sample)
            batch_data.append([d])
            batch_label.append(label)
            if len(batch_data) == batch_size:
                yield np.vstack(batch_data), np.vstack(batch_label).astype(np.int64)
                batch_data, batch_label = [], []

    return reader


def test_reader(data_path, batch_size):
    def reader():
        readData = ReadData(data_path)
        keys = readData.get_keys()
        keys = list(keys)

        batch_data, batch_label = [], []
        for key in keys:
            data = readData.get_data(key)
            assert (data is not None)
            label = readData.get_label(key)
            assert (label is not None)
            sample = (data, label)
            d, label = mapper(sample)
            batch_data.append([d])
            batch_label.append(label)
            if len(batch_data) == batch_size:
                yield np.vstack(batch_data), np.vstack(batch_label).astype(np.int64)
                batch_data, batch_label = [], []

    return reader
```

## 训练

接着就可以开始训练模型了，创建 `train.py`。我们搭建简单的卷积神经网络，如果音频种类非常多，可以适当使用更大的卷积神经网络模型。通过把音频数据转换成梅尔频谱，数据的shape也相当于灰度图，所以为 `(1, 128, 128)`。然后定义优化方法和获取训练和测试数据。要注意 `CLASS_DIM`参数的值，这个是类别的数量，要根据你数据集中的分类数量来修改。

```python
import reader
import paddle.fluid as fluid

# 保存预测模型路径
save_path = 'models/'
# 类别总数
CLASS_DIM = 10

# 定义输入层
image = fluid.data(name='image', shape=[None, 1, 128, 128], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')


# 卷积神经网络
def cnn(input, class_dim):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=20,
                                filter_size=5,
                                act='relu')

    conv2 = fluid.layers.conv2d(input=conv1,
                                num_filters=50,
                                filter_size=5,
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max')

    bn = fluid.layers.batch_norm(pool1)
    flatten = fluid.layers.flatten(bn)
    f1 = fluid.layers.fc(input=flatten, size=128, act='relu')
    f2 = fluid.layers.fc(input=f1, size=class_dim, act='softmax')
    return f2


# 获取网络模型
model = cnn(image, CLASS_DIM)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = reader.train_reader('dataset/train', batch_size=32)
test_reader = reader.test_reader('dataset/test', batch_size=32)

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

最后执行训练，每100个batch打印一次训练日志，训练一轮之后执行测试和保存模型，在测试时，把每个batch的输出都统计，最后求平均值。保存的模型为预测模型，方便之后的预测使用。

```python
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed={image.name: data[0], label.name: data[1]},
                                        fetch_list=[avg_cost, acc])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed={image.name: data[0], label.name: data[1]},
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存预测模型
    fluid.io.save_inference_model(dirname=save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)

```

# 预测

在训练结束之后，我们得到了一个预测模型，有了预测模型，执行预测非常方便。我们使用这个模型预测音频，输入的音频不能小于2.97秒，也不能太长，因为之截取前面的2.97秒的音频进行预测。在执行预测之前，需要把音频转换为梅尔频谱数据，并把数据shape转换为(1, 1, 128, 128)，第一个为输入数据的batch大小，如果想多个音频一起数据，可以把他们存放在list中一起预测。最后输出的结果即为预测概率最大的标签。

```python
import librosa
import numpy as np
import paddle.fluid as fluid

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'models/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 读取音频数据
def load_data(data_path):
    y1, sr1 = librosa.load(data_path, duration=2.97)
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1).astype(np.float32)
    ps = ps[np.newaxis, np.newaxis, ...]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    # 执行预测
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: data},
                     fetch_list=target_var)
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][0][-1]
    return lab


if __name__ == '__main__':
    # 要预测的音频文件
    path = 'dataset/UrbanSound8K/audio/fold8/193699-2-0-46.wav'
    label = infer(path)
    print('音频：%s 的预测结果标签为：%d' % (path, label))
```

# 其他

为了方便读取录制数据和制作数据集，这里提供了两个程序，首先是 `record_audio.py`，这个用于录制音频，录制的音频帧率为44100，通道为1，16bit。

```python
import pyaudio
import wave
import uuid
from tqdm import tqdm
import os

s = input('请输入你计划录音多少秒：')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = int(s)
WAVE_OUTPUT_FILENAME = "save_audio/%s.wav" % str(uuid.uuid1()).replace('-', '')

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始录音, 请说话......")

frames = []

for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音已结束!")

stream.stop_stream()
stream.close()
p.terminate()

if not os.path.exists('save_audio'):
    os.makedirs('save_audio')

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print('文件保存在：%s' % WAVE_OUTPUT_FILENAME)
os.system('pause')
```

创建 `crop_audio.py`，在训练是只是裁剪前面的2.97秒的音频，所以我们要把录制的硬盘安装每3秒裁剪一段，把裁剪后音频存放在音频名称命名的文件夹中。最后把这些文件按照训练数据的要求创建数据列表和训练数据。

```python
import os
import uuid
import wave
from pydub import AudioSegment


# 按秒截取音频
def get_part_wav(sound, start_time, end_time, part_wav_path):
    save_path = os.path.dirname(part_wav_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start_time = int(start_time) * 1000
    end_time = int(end_time) * 1000
    word = sound[start_time:end_time]
    word.export(part_wav_path, format="wav")


def crop_wav(path, crop_len):
    for src_wav_path in os.listdir(path):
        wave_path = os.path.join(path, src_wav_path)
        print(wave_path[-4:])
        if wave_path[-4:] != '.wav':
            continue
        file = wave.open(wave_path)
        # 帧总数
        a = file.getparams().nframes
        # 采样频率
        f = file.getparams().framerate
        # 获取音频时间长度
        t = int(a / f)
        print('总时长为 %d s' % t)
        # 读取语音
        sound = AudioSegment.from_wav(wave_path)
        for start_time in range(0, t, crop_len):
            save_path = os.path.join(path, os.path.basename(wave_path)[:-4], str(uuid.uuid1()) + '.wav')
            get_part_wav(sound, start_time, start_time + crop_len, save_path)


if __name__ == '__main__':
    crop_len = 3
    crop_wav('save_audio', crop_len)
```

创建 `infer_record.py`，这个程序是用来不断进行录音识别，因为识别的时间比较短，所以我们可以大致理解为这个程序在实时录音识别。通过这个应该我们可以做一些比较有趣的事情，比如把麦克风放在小鸟经常来的地方，通过实时录音识别，一旦识别到有鸟叫的声音，如果你的数据集足够强大，有每种鸟叫的声音数据集，这样你还能准确识别是那种鸟叫。如果识别到目标鸟类，就启动程序，例如拍照等等。

```python
import wave
import librosa
import numpy as np
import pyaudio
import paddle.fluid as fluid

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'models/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "infer_audio.wav"

# 打开录音
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# 读取音频数据
def load_data(data_path):
    y1, sr1 = librosa.load(data_path, duration=2.97)
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1).astype(np.float32)
    ps = ps[np.newaxis, np.newaxis, ...]
    return ps


# 获取录音数据
def record_audio():
    print("开始录音......")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音已结束!")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


# 预测
def infer(audio_data):
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: audio_data},
                     fetch_list=target_var)
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][0][-1]
    return lab


if __name__ == '__main__':
    try:
        while True:
            # 加载数据
            data = load_data(record_audio())

            # 获取预测结果
            label = infer(data)
            print('预测的标签为：%d' % label)
    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
```

**Github地址：**[https://github.com/yeyupiaoling/AudioClassification_PaddlePaddle](https://github.com/yeyupiaoling/AudioClassification_PaddlePaddle)
