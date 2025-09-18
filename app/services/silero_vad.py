
import os
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AudioChunkResult:
    files: List[str]               # 保存された chunk のファイルパス
    timestamps: List[Dict[str, int]]  # VAD が返す speech_timestamps

# Silero VAD モデルのロード
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)


(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def split_audio_with_vad(input_wav: str, sounds_dir: str = "sounds", sr: int = 16000):
    """
    Silero VADで音声を分割し、/sounds/<timestamp>/chunk_{i}.wav に保存する関数
    Args:
        input_wav (str): 入力音声ファイルのパス (wav推奨, sr=16kHz)
        sounds_dir (str): 保存ディレクトリ (default: sounds)
        sr (int): サンプリングレート (default: 16000)
    Returns:
        List[str]: 保存された chunk のファイルパス一覧
    """
    # 出力先ディレクトリを作成（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(sounds_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # 音声をロード
    wav = read_audio(input_wav, sampling_rate=sr)

    # 音声区間を抽出
    speech_timestamps = get_speech_timestamps(wav, model_vad, sampling_rate=sr)

    saved_files = []
    for i, ts in enumerate(speech_timestamps):
        chunk_wav = collect_chunks([ts], wav)  # 区間の切り出し
        out_path = os.path.join(out_dir, f"chunk_{i}.wav")
        save_audio(out_path, chunk_wav, sampling_rate=sr)
        saved_files.append(out_path)
        
    return AudioChunkResult(files=saved_files, timestamps=speech_timestamps)
