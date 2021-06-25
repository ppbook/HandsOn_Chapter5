!pip install --upgrade librosa

import librosa

# 観測信号の読み込み
x, x_sr = librosa.load(librosa.example('vibeace'), sr=22050, mono=True)

# 短時間フーリエ変換
X = librosa.stft(x, n_fft=2048, hop_length=512)
X_mag, X_phase = librosa.magphase(X)

import scipy

# メディアンフィルタ
H = scipy.ndimage.median_filter(X_mag, size=(1, 31))
P = scipy.ndimage.median_filter(X_mag, size=(31, 1))

# 調波音のウィナーフィルタ
M_H = H**2 / (H**2 + P**2)
H_wiener = M_H * X_mag

# 打楽器音のウィナーフィルタ
M_P = P**2 / (H**2 + P**2)
P_wiener = M_P * X_mag

# 逆短時間フーリエ変換
h = librosa.istft(H_wiener * X_phase)
p = librosa.istft(P_wiener * X_phase)

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 調波音のスペクトログラムを表示
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(H_wiener * X_phase)), x_axis='s', y_axis='hz', cmap='gray') 
plt.title('harmonic spectrogram')

# 打楽器音のスペクトログラムを表示
plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(P_wiener * X_phase)), x_axis='s', cmap='gray')
plt.title('percussive spectrogram')

plt.tight_layout()
plt.show()

from google.colab import drive
import soundfile as sf

# Google Driveをマウント
drive.mount('/content/drive')

# 調波音をwav形式で書き出し
sf.write('/content/drive/My Drive/harm.wav', h, x_sr)

# 打楽器音をwav形式で書き出し
sf.write('/content/drive/My Drive/perc.wav', p, x_sr)
