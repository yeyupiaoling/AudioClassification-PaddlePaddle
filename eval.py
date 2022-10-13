import argparse
import functools

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import accuracy
from sklearn.metrics import confusion_matrix

from modules.ecapa_tdnn import EcapaTdnn
from data_utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('audio_duration',   float,  3,                        '评估的音频长度，单位秒')
add_arg('num_class',        int,    10,                       '分类的类别数量')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',  str,    'dataset/label_list.txt', '标签列表路径')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('model_path',       str,    'output/models/model.pdparams',  '模型保存的路径')
args = parser.parse_args()


def evaluate():
    # 获取评估数据
    eval_dataset = CustomDataset(args.test_list_path,
                                 feature_method=args.feature_method,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=args.audio_duration)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn,
                             num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]
    # 获取模型
    if args.use_model == 'ecapa_tdnn':
        model = EcapaTdnn(num_class=args.num_class, input_size=eval_dataset.input_size)
    else:
        raise Exception(f'{args.use_model} 模型不存在!')
    paddle.summary(model, input_size=(1, eval_dataset.input_size, 98))
    # 加载模型
    model.set_state_dict(paddle.load(args.model_path))
    model.eval()
    # 开始评估
    accuracies, preds, labels = [], [], []
    for batch_id, (audio, label, audio_lens) in enumerate(eval_loader()):
        output = model(audio, audio_lens)
        label1 = paddle.reshape(label, shape=(-1, 1))
        acc = accuracy(input=output, label=label1)
        # 模型预测标签
        pred = paddle.argsort(output, descending=True)[:, 0].numpy().tolist()
        preds.extend(pred)
        # 真实标签
        labels.extend(label.numpy().tolist())
        # 准确率
        accuracies.append(acc.numpy()[0])
    acc = float(sum(accuracies) / len(accuracies))
    cm = confusion_matrix(labels, preds)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # 精确率
    precision = TP / (TP + FP + 1e-6)
    # 召回率
    recall = TP / (TP + FN + 1e-6)
    f1_score = (2 * precision * recall) / (precision + recall)
    print('分类准确率: {:.4f}, F1-Score:: {:.4f}'.format(acc, np.mean(f1_score)))
    plot_confusion_matrix(cm=cm, save_path='output/log/混淆矩阵_eval.png', class_labels=class_labels)


if __name__ == '__main__':
    print_arguments(args)
    evaluate()
