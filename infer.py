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
