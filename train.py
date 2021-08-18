import argparse
import functools
import os
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.metric import accuracy
from paddle.static import InputSpec
from resnet import resnet34
from reader import CustomDataset
from utility import add_arguments, print_arguments

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
add_arg('save_model',       str,    'models/',                '模型保存的路径')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def test(model, test_loader):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader()):
        output = model(spec_mag)
        label = paddle.reshape(label, shape=(-1, 1))
        acc = accuracy(input=output, label=label)
        accuracies.append(acc.numpy()[0])
    model.train()
    return float(sum(accuracies) / len(accuracies))


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

    # 获取模型
    model = resnet34(num_classes=args.num_classes)
    if dist.get_rank() == 0:
        paddle.summary(model, input_size=input_shape)

    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        model = paddle.DataParallel(model)

    # 初始化epoch数
    last_epoch = 0
    # 学习率衰减
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=10, gamma=0.1, verbose=True)
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(5e-4))

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
            acc = test(model, test_loader)
            print('='*70)
            print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
            print('='*70)
            # 保存预测模型
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            paddle.jit.save(layer=model,
                            path=os.path.join(args.save_model, 'model'),
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
