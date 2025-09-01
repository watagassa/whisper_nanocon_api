import whisper

def main():
    # モデルを読み込み（base以外も選べる: tiny, small, medium, large）
    model = whisper.load_model("base")

    # ユーザにmp3ファイルのパスを入力してもらう
    file_path = input("音声ファイル(mp3)のパスを入力してください: ")

    try:
        # 音声を文字起こし
        result = model.transcribe(file_path, language="ja")

        # 結果を表示
        print("\n--- 認識結果 ---")
        print(result["text"])

        # セグメントごとの詳細も出したい場合
        print("\n--- セグメント詳細 ---")
        for seg in result["segments"]:
            print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
