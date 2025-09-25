import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

def preprocess_audio(input_path: str, output_path: str):
    # --- 1. 音声読み込み（16kHzに変換）---
    y, sr = librosa.load(input_path, sr=16000)

    # --- 2. 無音チェック ---
    if np.max(np.abs(y)) < 1e-6:
        print("⚠ 無音のため処理をスキップします")
        sf.write(output_path, y, sr, subtype="PCM_16")
        return output_path

    # --- 3. ノイズ除去（安定化設定） ---
    reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)

    # --- 4. 音量正規化（RMSベース） ---
    rms = (reduced_noise**2).mean()**0.5
    if rms < 1e-6:
        normalized = reduced_noise
    else:
        target_rms = 0.1
        normalized = reduced_noise * (target_rms / rms)

    # --- 5. 書き出し（16bit PCM） ---
    sf.write(output_path, normalized, sr, subtype="PCM_16")

    return output_path
