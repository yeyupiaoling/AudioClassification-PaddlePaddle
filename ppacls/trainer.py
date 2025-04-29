import os
import platform
import sys
import time
import uuid
from datetime import timedelta

import numpy as np
import paddle
import yaml
from paddle import summary
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddle.metric import accuracy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from visualdl import LogWriter

from loguru import logger

from ppacls.data_utils.collate_fn import collate_fn
from ppacls.data_utils.featurizer import AudioFeaturizer
from ppacls.data_utils.reader import PPAClsDataset
from ppacls.models import build_model
from ppacls.optimizer import build_lr_scheduler, build_optimizer
from ppacls.utils.checkpoint import load_pretrained, load_checkpoint, save_checkpoint
from ppacls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments, convert_string_based_on_type


class PPAClsTrainer(object):
    def __init__(self,
                 configs,
                 use_gpu=True,
                 data_augment_configs=None,
                 num_class=None,
                 overwrites=None,
                 log_level="info"):
        """声音分类训练工具类

        :param configs: 配置文件路径，或者模型名称，如果是模型名称则会使用默认的配置文件
        :param use_gpu: 是否使用GPU训练模型
        :param data_augment_configs: 数据增强配置字典或者其文件路径
        :param num_class: 分类大小，对应配置文件中的model_conf.model_args.num_class
        :param overwrites: 覆盖配置文件中的参数，比如"train_conf.max_epoch=100"，多个用逗号隔开
        :param log_level: 打印的日志等级，可选值有："debug", "info", "warning", "error"
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.use_gpu = use_gpu
        self.log_level = log_level.upper()
        logger.remove()
        logger.add(sink=sys.stdout, level=self.log_level)
        # 读取配置文件
        if isinstance(configs, str):
            # 获取当前程序绝对路径
            absolute_path = os.path.dirname(__file__)
            # 获取默认配置文件路径
            config_path = os.path.join(absolute_path, f"configs/{configs}.yml")
            configs = config_path if os.path.exists(config_path) else configs
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.configs = dict_to_object(configs)
        if num_class is not None:
            self.configs.model_conf.model_args.num_class = num_class
        # 覆盖配置文件中的参数
        if overwrites:
            overwrites = overwrites.split(",")
            for overwrite in overwrites:
                keys, value = overwrite.strip().split("=")
                attrs = keys.split('.')
                current_level = self.configs
                for attr in attrs[:-1]:
                    current_level = getattr(current_level, attr)
                before_value = getattr(current_level, attrs[-1])
                setattr(current_level, attrs[-1], convert_string_based_on_type(before_value, value))
        # 打印配置信息
        print_arguments(configs=self.configs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        # 读取数据增强配置文件
        if isinstance(data_augment_configs, str):
            with open(data_augment_configs, 'r', encoding='utf-8') as f:
                data_augment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=data_augment_configs, title='数据增强配置')
        self.data_augment_configs = dict_to_object(data_augment_configs)
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False):
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        sampler_args = self.configs.dataset_conf.get('sampler', {})
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        if is_train:
            self.train_dataset = PPAClsDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               audio_featurizer=self.audio_featurizer,
                                               aug_conf=self.data_augment_configs,
                                               mode='train',
                                               **dataset_args)
            train_sampler = BatchSampler(dataset=self.train_dataset, **sampler_args)
            if paddle.distributed.get_world_size() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedBatchSampler(dataset=self.train_dataset, **sampler_args)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           batch_sampler=train_sampler,
                                           **data_loader_args)
        # 获取测试数据
        data_loader_args.drop_last = False
        dataset_args.max_duration = self.configs.dataset_conf.eval_conf.max_duration
        data_loader_args.batch_size = self.configs.dataset_conf.eval_conf.batch_size
        self.test_dataset = PPAClsDataset(data_list_path=self.configs.dataset_conf.test_list,
                                          audio_featurizer=self.audio_featurizer,
                                          mode='eval',
                                          **dataset_args)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)

    # 提取特征保存文件
    def extract_features(self, save_dir='dataset/features', max_duration=100):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        dataset_args = self.configs.dataset_conf.get('dataset', {})
        dataset_args.max_duration = max_duration
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        data_loader_args.drop_last = False
        for data_list in [self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]:
            test_dataset = PPAClsDataset(data_list_path=data_list,
                                         audio_featurizer=self.audio_featurizer,
                                         mode='extract_feature',
                                         **dataset_args)
            test_loader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     **data_loader_args)
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for features, labels, input_lens in tqdm(test_loader()):
                    for i in range(len(features)):
                        feature, label, input_len = features[i], labels[i], input_lens[i]
                        feature = feature.numpy()[:input_len]
                        label = int(label)
                        save_path = os.path.join(save_dir, str(label),
                                                 f'{str(uuid.uuid4())}.npy').replace('\\', '/')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        np.save(save_path, feature)
                        f.write(f'{save_path}\t{label}\n')
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')

    def __setup_model(self, input_size, is_train=False):
        # 自动获取列表数量
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)
        # 获取模型
        self.model = build_model(input_size=input_size, configs=self.configs)
        if self.log_level == "DEBUG" or self.log_level == "INFO":
            # 打印模型信息，98是长度，这个取决于输入的音频长度
            summary(self.model, (1, 98, input_size))
        # print(self.model)
        # 获取损失函数
        label_smoothing = self.configs.train_conf.get('label_smoothing', 0.0)
        self.loss = paddle.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if is_train:
            if self.configs.train_conf.enable_amp:
                # 自动混合精度训练，逻辑2，定义GradScaler
                self.amp_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            # 学习率衰减函数
            self.scheduler = build_lr_scheduler(step_per_epoch=len(self.train_loader), configs=self.configs)
            # 获取优化方法
            self.optimizer = build_optimizer(parameters=self.model.parameters(), learning_rate=self.scheduler,
                                             configs=self.configs)

    def __train_epoch(self, epoch_id, local_rank, writer):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        for batch_id, (features, label, input_lens) in enumerate(self.train_loader()):
            if self.stop_train: break
            # 执行模型计算，是否开启自动混合精度
            with paddle.amp.auto_cast(enable=self.configs.train_conf.enable_amp, level='O1'):
                output = self.model(features)
            # 计算损失值
            los = self.loss(output, label)
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # 更新参数（参数梯度先除系数loss_scaling再更新参数）
                self.amp_scaler.step(self.optimizer)
                # 基于动态loss_scaling策略更新loss_scaling系数
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(float(acc))
            loss_sum.append(float(los))
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.sampler.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_lr():>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_lr(), self.train_log_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_log_step += 1
            self.scheduler.step()
            start = time.time()

    def train(self,
              save_model_path='models/',
              log_dir='log/',
              max_epoch=None,
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param log_dir: 保存VisualDL日志文件的路径
        :param max_epoch: 最大训练轮数，对应配置文件中的train_conf.max_epoch
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        paddle.seed(1000)
        # 获取有多少张显卡训练
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir=log_dir)

        if nranks > 1 and self.use_gpu:
            # 初始化Fleet环境
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)

        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)
        # 加载预训练模型
        self.model = load_pretrained(model=self.model, pretrained_model=pretrained_model)
        # 加载恢复模型
        self.model, self.optimizer, self.amp_scaler, self.scheduler, last_epoch, best_acc = \
            load_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, scheduler=self.scheduler, step_epoch=len(self.train_loader),
                            save_model_path=save_model_path, resume_model=resume_model)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        if max_epoch is not None:
            self.configs.train_conf.max_epoch = max_epoch
        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_loss, self.eval_acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                self.test_log_step += 1
                self.model.train()
                # 保存最优模型
                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                    amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                    accuracy=self.eval_acc, best_model=True)
                # 保存模型
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                accuracy=self.eval_acc)

    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            missing_keys, unexpected_keys = self.model.set_state_dict(model_state_dict)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                logger.warning(f'模型加载部分失败，请检查模型是否匹配\n'
                               f'missing_keys: {missing_keys}\nunexpected_keys: {unexpected_keys}')
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with paddle.no_grad():
            for batch_id, (features, label, input_lens) in enumerate(tqdm(self.test_loader(), desc='执行评估')):
                if self.stop_eval: break
                output = eval_model(features)
                los = self.loss(output, label)
                # 计算准确率
                label = paddle.reshape(label, shape=(-1, 1))
                acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
                accuracies.append(float(acc))
                losses.append(float(los))
                # 模型预测标签
                pred = paddle.argsort(output, descending=True)[:, 0].numpy().tolist()
                preds.extend(pred)
                # 真实标签
                labels.extend(label.numpy().tolist())
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1
        # 保存混合矩阵
        if save_matrix_path is not None:
            try:
                cm = confusion_matrix(labels, preds)
                plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                      class_labels=self.class_labels)
            except Exception as e:
                logger.error(f'保存混淆矩阵失败：{e}')
        self.model.train()
        return loss, acc

    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = paddle.load(resume_model)
        self.model.set_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_dir = os.path.join(save_model_path,
                                       f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                       'infer')
        os.makedirs(infer_model_dir, exist_ok=True)
        infer_model_path = os.path.join(infer_model_dir, 'model')
        paddle.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
