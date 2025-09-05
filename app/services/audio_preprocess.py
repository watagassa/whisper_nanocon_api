import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import numpy as np

def preprocess_audio(input_path: str, output_path: str):
    # --- 1. 音声読み込み（librosaで16kHzに変換）---
    y, sr = librosa.load(input_path, sr=16000)

    # --- 2. ノイズ除去 ---
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    # reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)


    # --- 3. 音量正規化（RMS基準で調整）---
    rms = (reduced_noise**2).mean()**0.5
    target_rms = 0.1  # 好みで調整
    normalized = reduced_noise * (target_rms / (rms + 1e-6))
    
    #     # --- 3. ピーク正規化 ---
    # peak = np.max(np.abs(reduced_noise))
    # if peak > 0:
    #     normalized = reduced_noise / peak * 0.99  # 最大振幅を 0.99 に
    # else:
    #     normalized = reduced_noise  # 無音の場合はそのまま

    # # --- 4. 書き出し ---
    sf.write(output_path, normalized, sr)

    return output_path
