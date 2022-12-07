import os

import numpy as np
import paddle

from ppacls import SUPPORT_MODEL
from ppacls.data_utils.audio import AudioSegment
from ppacls.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppacls.models.ecapa_tdnn import EcapaTdnn
from ppacls.models.panns import CNN6, CNN10, CNN14
from ppacls.utils.logger import setup_logger
from ppacls.utils.utils import dict_to_object

logger = setup_logger(__name__)


class PPAClsPredictor:
    def __init__(self,
                 configs,
                 model_path='models/ecapa_tdnn_spectrogram/best_model/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置参数
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self._audio_featurizer = AudioFeaturizer(feature_conf=self.configs.feature_conf, **self.configs.preprocess_conf)
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 获取模型
        if self.configs.use_model == 'ecapa_tdnn':
            self.predictor = EcapaTdnn(input_size=self._audio_featurizer.feature_dim,
                                       num_class=self.configs.dataset_conf.num_class,
                                       **self.configs.model_conf)
        elif self.configs.use_model == 'panns_cnn6':
            self.model = CNN6(input_size=self._audio_featurizer.feature_dim,
                              num_class=self.configs.dataset_conf.num_class,
                              **self.configs.model_conf)
        elif self.configs.use_model == 'panns_cnn10':
            self.model = CNN10(input_size=self._audio_featurizer.feature_dim,
                               num_class=self.configs.dataset_conf.num_class,
                               **self.configs.model_conf)
        elif self.configs.use_model == 'panns_cnn14':
            self.model = CNN14(input_size=self._audio_featurizer.feature_dim,
                               num_class=self.configs.dataset_conf.num_class,
                               **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pdparams')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        self.predictor.set_state_dict(paddle.load(model_path))
        print(f"成功加载模型参数：{model_path}")
        self.predictor.eval()
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]

    # 预测一个音频的特征
    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频

        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            input_data = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            input_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            input_data = AudioSegment.from_wave_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        audio_feature = self._audio_featurizer.featurize(input_data)
        input_data = paddle.to_tensor(audio_feature, dtype=paddle.float32).unsqueeze(0)
        data_length = paddle.to_tensor([input_data.shape[1]], dtype=paddle.int64)
        # 执行预测
        output = self.predictor(input_data, data_length)
        result = paddle.nn.functional.softmax(output).numpy()[0]
        # 最大概率的label
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data):
        """预测一批音频

        :param audios_data: 经过预处理的一批数据
        :return: 结果标签和对应的得分
        """
        data_length = paddle.to_tensor([a.shape[0] for a in audios_data], dtype=paddle.int64)
        # 找出音频长度最长的
        batch = sorted(audios_data, key=lambda a: a.shape[0], reverse=True)
        freq_size = batch[0].shape[1]
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
        for i, sample in enumerate(batch):
            seq_length = sample.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[i, :seq_length, :] = sample[:, :]
        audios_data = paddle.to_tensor(inputs, dtype=paddle.float32)
        # 执行预测
        output = self.predictor(audios_data, data_length)
        results = paddle.nn.functional.softmax(output).numpy()
        labels, scores = [], []
        for result in results:
            lab = np.argsort(result)[-1]
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))
        return labels, scores
