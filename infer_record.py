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
