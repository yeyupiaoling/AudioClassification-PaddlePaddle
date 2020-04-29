import os
import struct
import uuid
import librosa
import numpy as np
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
            y1, sr1 = librosa.load(path, sr=16000, duration=2.04)
            ps = librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length=256).reshape(-1).tolist()
            if len(ps) != 128 * 128: continue
            data = struct.pack('%sd' % len(ps), *ps)
            # 写入对应的数据
            key = str(uuid.uuid1())
            writer.add_data(key, data)
            writer.add_label('\t'.join([key, label.replace('\n', '')]))
        except Exception as e:
            print(e)


# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    for i in range(len(audios)):
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound)
            t = librosa.get_duration(filename=sound_path)
            # 过滤小于2.1秒的音频
            if t >= 2.1:
                if sound_sum % 100 == 0:
                    f_test.write('%s\t%d\n' % (sound_path, i))
                else:
                    f_train.write('%s\t%d\n' % (sound_path, i))
                sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))

    f_test.close()
    f_train.close()


# 裁剪静音片段
def crop_silence(audios_path):
    print("正在裁剪静音片段...")
    for root, dirs, files in os.walk(audios_path, topdown=False):
        for name in files:
            audio_path = os.path.join(root, name)
            wav, sr = librosa.load(audio_path)

            intervals = librosa.effects.split(wav, top_db=20)
            wav_output = []
            for sliced in intervals:
                wav_output.extend(wav[sliced[0]:sliced[1]])
            wav_output = np.array(wav_output)
            librosa.output.write_wav(audio_path, wav_output, sr)

    print("裁剪完成！")


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
    crop_silence('dataset/audio')
    # get_urbansound8k_list('dataset', 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
    get_data_list('dataset/audio', 'dataset')
    convert_data('dataset/train_list.txt', 'dataset/train')
    convert_data('dataset/test_list.txt', 'dataset/test')
