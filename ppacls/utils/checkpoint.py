import json
import os
import shutil

import paddle

from ppacls import __version__
from ppacls.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_pretrained(model, pretrained_model):
    """加载预训练模型

    :param model: 使用的模型
    :param pretrained_model: 预训练模型路径
    """
    # 加载预训练模型
    if pretrained_model is None: return model
    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
    model_dict = model.state_dict()
    model_state_dict = paddle.load(pretrained_model)
    # 过滤不存在的参数
    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if list(weight.shape) != list(model_state_dict[name].shape):
                logger.warning('{} not used, shape {} unmatched with {} in model.'.
                               format(name, list(model_state_dict[name].shape), list(weight.shape)))
                model_state_dict.pop(name, None)
        else:
            logger.warning('Lack weight: {}'.format(name))
    # 加载权重
    missing_keys, unexpected_keys = model.set_state_dict(model_state_dict)
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))
    logger.info('成功加载预训练模型：{}'.format(pretrained_model))
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,
                    step_epoch, save_model_path, resume_model):
    """加载模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param scheduler: 使用的学习率调整策略
    :param step_epoch: 每个epoch的step数量
    :param save_model_path: 模型保存路径
    :param resume_model: 恢复训练的模型路径
    """
    last_epoch1 = -1
    accuracy1 = 0.

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        state_dict = paddle.load(os.path.join(model_path, 'model.pdparams'))
        model.set_state_dict(state_dict)
        optimizer.set_state_dict(paddle.load(os.path.join(model_path, 'optimizer.pdopt')))
        # 自动混合精度参数
        if amp_scaler is not None and os.path.exists(os.path.join(model_path, 'scaler.pdparams')):
            amp_scaler.set_state_dict(paddle.load(os.path.join(model_path, 'scaler.pdparams')))
        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            last_epoch = json_data['last_epoch'] - 1
            accuracy = json_data['accuracy']
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        optimizer.step()
        [scheduler.step() for _ in range(last_epoch * step_epoch)]
        return last_epoch, accuracy

    # 获取最后一个保存的模型
    save_feature_method = configs.preprocess_conf.feature_method
    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')
    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                    and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
        if resume_model is not None:
            last_epoch1, accuracy1 = load_model(resume_model)
        else:
            try:
                # 自动获取最新保存的模型
                last_epoch1, accuracy1 = load_model(last_model_dir)
            except Exception as e:
                logger.warning(f'尝试自动恢复最新模型失败，错误信息：{e}')
    return model, optimizer, amp_scaler, scheduler, last_epoch1, accuracy1


# 保存模型
def save_checkpoint(configs, model, optimizer, amp_scaler, save_model_path, epoch_id,
                    accuracy=0., best_model=False):
    """保存模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param save_model_path: 模型保存路径
    :param epoch_id: 当前epoch
    :param accuracy: 当前准确率
    :param best_model: 是否为最佳模型
    """
    # 保存模型的路径
    save_feature_method = configs.preprocess_conf.feature_method
    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'epoch_{}'.format(epoch_id))
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    # 保存模型参数
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    # 自动混合精度参数
    if amp_scaler is not None:
        paddle.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pdparams'))
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        data = {"last_epoch": epoch_id, "accuracy": accuracy, "version": __version__,
                "model": configs.model_conf.model, "feature_method": save_feature_method}
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    if not best_model:
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}', 'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path,
                                      f'{configs.model_conf.model}_{save_feature_method}',
                                      'epoch_{}'.format(epoch_id - 3))
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}'.format(model_path))
