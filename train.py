import argparse
import functools
import os
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.metric import accuracy
from paddle.static import InputSpec
from sklearn.metrics import confusion_matrix

from utils.resnet import resnet34
from utils.reader import CustomDataset
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,    '0',                      '训练使用的GPU序号，使用英文逗号,隔开，如：0,1')
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 128, 128)',    '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'dataset/label_list.txt', '标签列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def evaluate(model, test_loader):
    model.eval()
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
        accuracies.append(acc.numpy()[0])
    model.train()
    acc = float(sum(accuracies) / len(accuracies))
    cm = confusion_matrix(labels, preds)
    return acc, cm


def train(args):
    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        dist.init_parallel_env()
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train', spec_len=input_shape[3])
    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[3])
    test_batch_sampler = paddle.io.BatchSampler(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]
    # 获取模型
    model = resnet34(num_classes=args.num_classes)
    if dist.get_rank() == 0:
        paddle.summary(model, input_size=input_shape)

    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        model = paddle.DataParallel(model)

    # 学习率衰减
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=10, gamma=0.8, verbose=True)
    # 设置优化方法
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(5e-4))
    # 恢复训练
    last_epoch = 0
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
        optimizer_state = paddle.load(os.path.join(args.resume, 'optimizer.pdopt'))
        optimizer.set_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = optimizer_state['LR_Scheduler']['last_epoch']
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        loss_sum = []
        accuracies = []
        for batch_id, (spec_mag, label) in enumerate(train_loader()):
            output = model(spec_mag)
            # 计算损失值
            los = loss(output, label)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(acc.numpy()[0])
            loss_sum.append(los)
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f' % (
                    datetime.now(), epoch, batch_id, len(train_loader), sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies)))
        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            acc, cm = evaluate(model, test_loader)
            plot_confusion_matrix(cm=cm, save_path=f'log/混淆矩阵_{epoch}.png', class_labels=class_labels, show=False)
            print('='*70)
            print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
            print('='*70)
            # 保存预测模型
            os.makedirs(args.save_model, exist_ok=True)
            paddle.save(model.state_dict(), os.path.join(args.save_model, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(args.save_model, 'optimizer.pdopt'))
            paddle.jit.save(layer=model,
                            path=os.path.join(args.save_model, 'inference'),
                            input_spec=[
                                InputSpec(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
                                          dtype='float32')])

        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    if len(args.gpus.split(',')) > 1:
        dist.spawn(train, args=(args,), gpus=args.gpus, nprocs=len(args.gpus.split(',')))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        train(args)
