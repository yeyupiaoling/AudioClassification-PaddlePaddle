import argparse
import functools

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import accuracy
from sklearn.metrics import confusion_matrix

from utils.reader import CustomDataset
from utils.resnet import resnet34
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 128, 128)',    '数据输入的形状')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'dataset/label_list.txt', '标签列表路径')
add_arg('model_path',       str,    'models/model.pdparams',  '模型保存的路径')
args = parser.parse_args()


def evaluate():
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取评估数据
    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[3])
    test_batch_sampler = paddle.io.BatchSampler(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]
    # 获取模型
    model = resnet34(num_classes=args.num_classes)
    paddle.summary(model, input_size=input_shape)
    # 加载模型
    model.set_state_dict(paddle.load(args.model_path))
    model.eval()
    # 开始评估
    accuracies, preds, labels = [], [], []
    for batch_id, (spec_mag, label) in enumerate(test_loader()):
        output = model(spec_mag)
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
    recall = TP / (TP + FN)
    print('分类准确率: {:.4f}, 平均精确率: {:.4f}, 平均召回率: {:.4f}'.format(acc, np.mean(precision), np.mean(recall)))
    plot_confusion_matrix(cm=cm, save_path='log/混淆矩阵_eval.png', class_labels=class_labels)


if __name__ == '__main__':
    print_arguments(args)
    evaluate()
