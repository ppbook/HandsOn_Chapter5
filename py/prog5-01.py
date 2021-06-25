!pip install --upgrade librosa
import librosa

# 音声の読み込み
y, sr = librosa.load(librosa.example('trumpet'), sr=22050, duration=5.0)

# リサンプリング
y_16k = librosa.resample(y, sr, 16000)

# データ数の確認
print(y.shape)
print(y_16k.shape)
