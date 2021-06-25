!pip install --upgrade librosa

import librosa
from google.colab import drive

# Google Driveをマウント
drive.mount('/content/drive')

# 観測信号の読み込み
x, x_sr = librosa.load('/content/drive/My Drive/observed.wav', sr=22050, mono=True)

# 観測信号を短時間フーリエ変換
X = librosa.stft(x, n_fft=2048, hop_length=512)

# 振幅スペクトルと位相スペクトルを抽出
X_mag, X_phase = librosa.magphase(X)

# 振幅スペクトルをdBスケールに変換
X_db = librosa.amplitude_to_db(X_mag)

import librosa.display
import matplotlib.pyplot as plt

# 観測信号のスペクトログラムをプロット
plt.figure(figsize=(8,3))
librosa.display.specshow(X_db, x_axis='s', y_axis='hz', cmap='gray')
plt.title('oberved spectrogram')
plt.show()

import numpy as np

# 雑音信号の読み込み
n, n_sr = librosa.load('/content/drive/My Drive/observed.wav', sr=22050, mono=True, offset=0.1, duration=1.0)

# 雑音信号を短時間フーリエ変換
N = librosa.stft(n, n_fft=2048, hop_length=512)

# 雑音信号の振幅スペクトルの平均値を求める
N_mag, _ = librosa.magphase(N)
N_mean = np.mean(N_mag, axis=1)

# 振幅スペクトルと振幅スペクトルの平均値のシェイプを確認
print(N_mag.shape)
print(N_mean.shape)

# 観測信号から雑音信号を減算
sub_rate = 1.0
S_mag = X_mag - sub_rate * N_mean.reshape(N_mean.shape[0], 1)

# 負数を0に置換
S_mag = np.maximum(0, S_mag)

import soundfile as sf

# 逆短時間フーリエ変換
s = librosa.istft(S_mag * X_phase)

# 音声信号をwav形式で書き出し
sf.write('/content/drive/My Drive/denoised_voice.wav', s, s_sr)

# 音声信号のスペクトログラムをプロット
plt.figure(figsize=(8, 3))
S_db = librosa.amplitude_to_db(S_mag)
librosa.display.specshow(S_db, x_axis='s', y_axis='hz', cmap='gray')
plt.title('voice spectrogram with spectral subtraction')
plt.show()

# パディング
pad_width = S_mag.shape[1] - N_mag.shape[1]
N_mag = np.pad(N_mag, ((0, 0), (0, pad_width)), 'symmetric')

# シェイプが同じであることを確認
print(S_mag.shape)
print(N_mag.shape)

# ウィナーフィルタ
H = S_mag**2 / (S_mag**2 + N_mag**2)
S_wiener = H * S_mag

# 逆短時間フーリエ変換
s_wiener = librosa.istft(S_wiener * X_phase)

# 音声信号をwav形式でファイルを書き出し
sf.write('/content/drive/My Drive/denoised_voice_wiener.wav', s_wiener, x_sr)

# 音声信号のスペクトログラムをプロット
plt.figure(figsize=(8, 3))
S_wiener_db = librosa.amplitude_to_db(S_wiener)
librosa.display.specshow(S_wiener_db, x_axis='s', y_axis='hz', cmap='gray')
plt.title('voice spectrogram with wiener filter')
plt.show()

