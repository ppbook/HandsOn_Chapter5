!pip install --upgrade librosa

import librosa
from google.colab import drive

# Google Driveをマウント
drive.mount('/content/drive/')

# 音声信号の読み込み
s, s_sr = librosa.load('/content/drive/My Drive/voice.wav', sr=22050, mono=True)

# 雑音信号の読み込み
n, n_sr = librosa.load('/content/drive/My Drive/noise.wav', sr=22050, mono=True)

# 音声信号と雑音信号の時間長を確認
print(librosa.get_duration(s, s_sr))
print(librosa.get_duration(n, n_sr))

# パディング
if len(s) < len(n):
  s = librosa.util.pad_center(s, len(n), mode='constant')
else:
  n = librosa.util.pad_center(n, len(s), mode='constant')

import librosa.display
import matplotlib.pyplot as plt

# 音声信号のプロット
plt.figure(figsize=(8,3))
plt.subplot(1, 2, 1)
librosa.display.waveplot(s, s_sr, color='grey')
plt.title('voice signal')
plt.ylabel('amplitude')
plt.ylim([-0.5, 0.5])

# 雑音信号のプロット
plt.subplot(1, 2, 2)
librosa.display.waveplot(n, n_sr, color='grey')
plt.title('noise signal')
plt.ylim([-0.5, 0.5])
plt.tick_params(labelleft=False)

plt.tight_layout()
plt.show()

# 観測信号（音声信号＋雑音信号）の生成
noise_rate = 0.5
x = s + noise_rate * n

# 観測信号のプロット
plt.figure(figsize=(8, 3))
librosa.display.waveplot(x, s_sr, color='grey')
plt.title('observed signal')
plt.ylabel('amplitude')
plt.show()

import soundfile as sf

# 観測信号をwav形式で書き出し
sf.write('/content/drive/My Drive/observed.wav', x, s_sr)
