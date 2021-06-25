!pip install --upgrade librosa
import librosa

# 音声信号の読み込み
y, _ = librosa.load(librosa.example('trumpet'), sr=22050)

# 音量の正規化
y_norm = y / abs(y).max()

# ピーク音量の確認
print(abs(y).max())
print(abs(y_norm).max())
