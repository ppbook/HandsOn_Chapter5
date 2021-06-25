!pip install --upgrade librosa
import librosa
import numpy as np

# ステレオ信号の読み込み
y, _ = librosa.load(librosa.example('brahms', hq=True), mono=False)

# チャンネルのモノラル化
y_mono = np.mean(y, axis=0)

# チャンネル数の確認
print(y.shape)
print(y_mono.shape)
